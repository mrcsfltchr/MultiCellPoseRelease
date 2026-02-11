import os
import json
import grpc
import argparse
import sys
from typing import Iterable, Iterator, Optional

from ..server import cellpose_remote_pb2 as pb2
from ..server import cellpose_remote_pb2_grpc as pb2_grpc


def make_channel(address: str, insecure: bool = True) -> grpc.Channel:
    if insecure:
        return grpc.insecure_channel(address)
    # TLS variant could load credentials here
    creds = grpc.ssl_channel_credentials()
    return grpc.secure_channel(address, creds)


def auth_metadata(token: Optional[str] = None, user_id: Optional[str] = None):
    tok = token or os.environ.get("CELLPOSE_CLIENT_TOKEN", "dev-token")
    md = [("authorization", f"Bearer {tok}")]
    if user_id:
        md.append(("x-user", user_id))
    return md


def health_check(channel: grpc.Channel) -> str:
    stub = pb2_grpc.HealthStub(channel) if hasattr(pb2_grpc, 'HealthStub') else None
    if stub is None:
        # generated stubs missing
        return "UNKNOWN"
    resp = stub.Check(pb2.HealthCheckRequest())
    return resp.status


def run_inference(channel: grpc.Channel, req: pb2.RunRequest, token: Optional[str] = None, user_id: Optional[str] = None) -> Iterator[pb2.JobUpdate]:
    stub = pb2_grpc.InferenceServiceStub(channel) if hasattr(pb2_grpc, 'InferenceServiceStub') else None
    if stub is None:
        raise RuntimeError("InferenceService stub not available (protos not generated)")
    md = auth_metadata(token, user_id=user_id)
    return stub.Run(req, metadata=md)


def upload_file(channel: grpc.Channel, project_id: str, file_path: str, token: Optional[str] = None, relpath: Optional[str] = None, user_id: Optional[str] = None, progress_cb=None) -> pb2.UploadReply:
    stub = pb2_grpc.FileServiceStub(channel) if hasattr(pb2_grpc, 'FileServiceStub') else None
    if stub is None:
        raise RuntimeError("FileService stub not available (protos not generated)")

    def chunk_iterator(project_id: str, file_path: str, relpath: str) -> Iterator[pb2.FileChunk]:
        total_size = None
        try:
            total_size = os.path.getsize(file_path)
        except Exception:
            total_size = None
        sent = 0
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                sent += len(chunk)
                if progress_cb is not None:
                    try:
                        cont = progress_cb(sent, total_size)
                        if cont is False:
                            return
                    except Exception:
                        pass
                yield pb2.FileChunk(project_id=project_id, relpath=relpath, data=chunk)

    md = auth_metadata(token, user_id=user_id)
    relpath = relpath or os.path.basename(file_path)
    response = stub.Upload(chunk_iterator(project_id, file_path, relpath), metadata=md)
    return response


def clear_user_train_jobs(channel: grpc.Channel, token: Optional[str] = None, user_id: Optional[str] = None) -> pb2.UploadReply:
    stub = pb2_grpc.FileServiceStub(channel) if hasattr(pb2_grpc, 'FileServiceStub') else None
    if stub is None:
        raise RuntimeError("FileService stub not available (protos not generated)")
    md = auth_metadata(token, user_id=user_id)

    def chunk_iterator() -> Iterator[pb2.FileChunk]:
        yield pb2.FileChunk(project_id="_admin", relpath="__clear_user_jobs__", data=b"")

    return stub.Upload(chunk_iterator(), metadata=md)


def download_file(channel: grpc.Channel, uri: str, output_path: str, token: Optional[str] = None, user_id: Optional[str] = None,
                  progress_cb=None, total_size: Optional[int] = None):
    stub = pb2_grpc.FileServiceStub(channel) if hasattr(pb2_grpc, 'FileServiceStub') else None
    if stub is None:
        raise RuntimeError("FileService stub not available (protos not generated)")
    
    md = auth_metadata(token, user_id=user_id)
    req = pb2.DownloadRequest(uri=uri)
    chunks = stub.Download(req, metadata=md)
    
    received = 0
    with open(output_path, "wb") as f:
        for chunk in chunks:
            if chunk.data:
                f.write(chunk.data)
                received += len(chunk.data)
                if progress_cb is not None:
                    try:
                        progress_cb(received, total_size)
                    except Exception:
                        pass


def list_files(channel: grpc.Channel, project_id: str, prefix: str = "", token: Optional[str] = None, user_id: Optional[str] = None):
    stub = pb2_grpc.FileServiceStub(channel) if hasattr(pb2_grpc, 'FileServiceStub') else None
    if stub is None:
        raise RuntimeError("FileService stub not available (protos not generated)")
    md = auth_metadata(token, user_id=user_id)
    uri = f"list://{project_id}"
    if prefix:
        from urllib.parse import quote
        uri = f"{uri}?prefix={quote(prefix)}"
    req = pb2.DownloadRequest(uri=uri)
    chunks = stub.Download(req, metadata=md)
    data = b""
    for chunk in chunks:
        data += chunk.data
    if not data:
        return []
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception:
        return []
    return payload.get("files", [])


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Cellpose gRPC client")
    parser.add_argument("--address", default="localhost:50051", help="Address of the gRPC server")
    parser.add_argument("--insecure", action="store_true", help="Use insecure channel")
    parser.add_argument("--token", default=None, help="Authentication token")

    subparsers = parser.add_subparsers(dest="command", required=True)

    health_parser = subparsers.add_parser("health", help="Check server health")

    run_parser = subparsers.add_parser("run", help="Run inference")
    run_parser.add_argument("--project-id", required=True, help="Project ID")
    run_parser.add_argument("--model-id", required=True, help="Model ID")
    run_parser.add_argument("uris", nargs="+", help="URIs of images to process")
    run_parser.add_argument("--diameter", type=float, help="Diameter")
    run_parser.add_argument("--cellprob_threshold", type=float, help="Cell probability threshold")
    run_parser.add_argument("--flow_threshold", type=float, help="Flow threshold")
    run_parser.add_argument("--do_3D", action="store_true", help="Do 3D segmentation")
    run_parser.add_argument("--niter", type=int, help="Number of iterations")
    run_parser.add_argument("--stitch_threshold", type=float, help="Stitch threshold")
    run_parser.add_argument("--anisotropy", type=float, help="Anisotropy")
    run_parser.add_argument("--flow3D_smooth", type=float, help="3D flow smoothing")
    run_parser.add_argument("--min_size", type=int, help="Minimum size")
    run_parser.add_argument("--max_size_fraction", type=float, help="Maximum size fraction")
    run_parser.add_argument("--normalize_params", type=str, help="Normalize parameters (JSON string)")
    run_parser.add_argument("--z_axis", type=int, help="Z axis")
    run_parser.add_argument("--channel_axis", type=int, help="Channel axis")


    upload_parser = subparsers.add_parser("upload", help="Upload a file")
    upload_parser.add_argument("--project-id", required=True, help="Project ID")
    upload_parser.add_argument("file", help="Path to the file to upload")

    download_parser = subparsers.add_parser("download", help="Download a file")
    download_parser.add_argument("uri", help="URI of the file to download")
    download_parser.add_argument("output", help="Path to save the downloaded file")

    args = parser.parse_args(argv)

    channel = make_channel(args.address, args.insecure)

    if args.command == "health":
        status = health_check(channel)
        print(f"Health status: {status}")
    elif args.command == "run":
        try:
            req = pb2.RunRequest(
                project_id=args.project_id, 
                uris=args.uris, 
                model_id=args.model_id
            )
            for key in ["diameter", "cellprob_threshold", "flow_threshold", "do_3D", "niter", "stitch_threshold", "anisotropy", "flow3D_smooth", "min_size", "max_size_fraction", "normalize_params", "z_axis", "channel_axis"]:
                if hasattr(args, key) and getattr(args, key) is not None:
                    setattr(req, key, getattr(args, key))

            updates = run_inference(channel, req, args.token)
            for update in updates:
                print(update)
        except grpc.RpcError as e:
            print(f"Error running inference: {e.details()}", file=sys.stderr)
            sys.exit(1)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "upload":
        try:
            reply = upload_file(channel, args.project_id, args.file, args.token)
            print(f"File uploaded to: {reply.uri}")
            print(f"Bytes received: {reply.bytes_received}")
        except grpc.RpcError as e:
            print(f"Error uploading file: {e.details()}", file=sys.stderr)
            sys.exit(1)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "download":
        try:
            download_file(channel, args.uri, args.output, args.token)
            print(f"File downloaded to: {args.output}")
        except grpc.RpcError as e:
            print(f"Error downloading file: {e.details()}", file=sys.stderr)
            sys.exit(1)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
