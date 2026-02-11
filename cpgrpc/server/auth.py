import os
import grpc
from typing import Any, Callable, Iterator, Optional


class AuthInterceptor(grpc.ServerInterceptor):
    """Simple bearer token auth via metadata.

    - Expects 'authorization: Bearer <token>'
    - Token is compared against env CELLPOSE_SERVER_TOKEN (default: 'dev-token')
    - Health.Check is allowed without auth
    """

    def __init__(self, token_env: str = "CELLPOSE_SERVER_TOKEN") -> None:
        self.token = os.environ.get(token_env, "dev-token")

    def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method or ""
        if method.endswith("/Check") and "Health" in method:
            return continuation(handler_call_details)

        # Extract metadata
        meta = dict(handler_call_details.invocation_metadata or [])
        auth = meta.get("authorization") or meta.get("authorization-bin")
        ok = False
        if isinstance(auth, bytes):
            try:
                auth = auth.decode("utf-8", errors="ignore")
            except Exception:
                auth = ""
        if isinstance(auth, str) and auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
            ok = (token == self.token)

        if not ok:
            def deny(request, context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "missing/invalid bearer token")

            # Map to the correct RPC type wrapper
            handler = continuation(handler_call_details)
            if handler is None:
                return None
            if handler.request_streaming and handler.response_streaming:
                return grpc.stream_stream_rpc_method_handler(lambda req_iter, ctx: deny(req_iter, ctx))
            if handler.request_streaming and not handler.response_streaming:
                return grpc.stream_unary_rpc_method_handler(lambda req_iter, ctx: deny(req_iter, ctx))
            if not handler.request_streaming and handler.response_streaming:
                return grpc.unary_stream_rpc_method_handler(lambda req, ctx: deny(req, ctx))
            return grpc.unary_unary_rpc_method_handler(lambda req, ctx: deny(req, ctx))

        return continuation(handler_call_details)


