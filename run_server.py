import signal
import logging
import sys
import argparse
from cpgrpc.server import services
from cellpose import remote_config
from cellpose import train as cellpose_train

_logger = logging.getLogger(__name__)


def _override_bind_host(bind: str, host: str) -> str:
    """Replace host portion of a host:port bind string."""
    if not host:
        return bind
    default_port = "50051"
    if ":" in bind:
        _, port = bind.rsplit(":", 1)
        port = port or default_port
    else:
        port = default_port
    return f"{host}:{port}"


def main():
    """Starts the cellpose gRPC server."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override bind host (e.g. 127.0.0.1 or 0.0.0.0).",
    )
    parser.add_argument("--train-debug", action="store_true", default=False)
    parser.add_argument("--train-debug-steps", type=int, default=3)
    args = parser.parse_args()
    cellpose_train.set_train_debug(args.train_debug, args.train_debug_steps)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
    cfg = remote_config.load_remote_config()
    bind = cfg.get("server_bind", "127.0.0.1:50051")
    if args.host:
        bind = _override_bind_host(bind, args.host)
    storage_root = cfg.get("server_storage_root", "./.cellpose_server_data")
    server = services.serve(bind=bind, storage_root=storage_root)
    _logger.info(f"Server started on {bind}")

    def _stop(*_):
        _logger.info("Stopping server...")
        server.stop(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        server.wait_for_termination()
    finally:
        server.stop(0)
        _logger.info("Server stopped.")


if __name__ == "__main__":
    main()
