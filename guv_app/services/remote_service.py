import grpc
import os
import subprocess
import sys
import time
import tempfile
import logging
from cpgrpc.client import client as grpc_client
from cpgrpc.server import cellpose_remote_pb2 as pb2
from cpgrpc.server import cellpose_remote_pb2_grpc as pb2_grpc
from guv_app.data_models.configs import RemoteConfig

class AuthInterceptor(grpc.UnaryUnaryClientInterceptor, grpc.UnaryStreamClientInterceptor, grpc.StreamUnaryClientInterceptor, grpc.StreamStreamClientInterceptor):
    """
    Client interceptor to inject the authentication token into gRPC metadata.
    """
    def __init__(self, token):
        self.token = token

    def _intercept_call(self, continuation, client_call_details, request_iterator):
        metadata = []
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)
        if self.token:
            metadata.append(("x-token", self.token))
        
        updated_details = client_call_details._replace(metadata=metadata)
        return continuation(updated_details, request_iterator)

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        return self._intercept_call(continuation, client_call_details, request_iterator)

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        return self._intercept_call(continuation, client_call_details, request_iterator)

_logger = logging.getLogger(__name__)

class RemoteConnectionService:
    """
    Service for managing the connection to the remote Cellpose server.
    Wraps gRPC calls for health checks, file transfer, and inference.
    """
    def __init__(self):
        self._config = RemoteConfig()
        self._channel = None
        self._stub_health = None
        self._stub_file = None
        self._stub_inference = None
        self._ssh_process = None
        self._ssh_askpass_path = None

    def __del__(self):
        self._stop_ssh_tunnel()

    def get_config(self):
        return self._config

    def _stop_ssh_tunnel(self):
        if self._ssh_process:
            try:
                self._ssh_process.terminate()
            except Exception:
                pass
            self._ssh_process = None
        
        if self._ssh_askpass_path and os.path.exists(self._ssh_askpass_path):
            try:
                os.remove(self._ssh_askpass_path)
            except Exception:
                pass
            self._ssh_askpass_path = None

    def _start_ssh_tunnel(self, credentials):
        host = credentials.get("host")
        user = credentials.get("username")
        password = credentials.get("password")
        port = credentials.get("port", 22)
        key_path = credentials.get("key_path")
        
        local_port = credentials.get("ssh_local_port", 50051)
        remote_port = credentials.get("ssh_remote_port", 50051)
        remote_bind = credentials.get("ssh_remote_bind", "127.0.0.1")
        # Force 127.0.0.1 if localhost is provided to avoid IPv6 resolution issues on remote
        if remote_bind == "localhost":
            remote_bind = "127.0.0.1"

        if not host or not user:
            return False

        target = f"{user}@{host}"
        
        args = [
            "ssh",
            "-N",
            "-L",
            f"{local_port}:{remote_bind}:{remote_port}",
            "-p",
            str(port),
        ]
        
        env = os.environ.copy()
        
        # If password is provided, prioritize it and ignore key to match cellpose behavior
        if password:
            args.extend(["-o", "PreferredAuthentications=keyboard-interactive,password"])
            args.extend(["-o", "KbdInteractiveAuthentication=yes"])
            args.extend(["-o", "PasswordAuthentication=yes"])
            args.extend(["-o", "PubkeyAuthentication=no"])
            args.extend(["-o", "NumberOfPasswordPrompts=1"])
            args.extend(["-o", "StrictHostKeyChecking=accept-new"])
            
            if sys.platform == "win32":
                askpass_cmd = tempfile.NamedTemporaryFile(delete=False, suffix=".cmd")
                askpass_cmd.write(b"@echo off\r\n")
                askpass_cmd.write(b"powershell -NoProfile -Command \"Write-Output $env:SSH_ASKPASS_PASSWORD\"\r\n")
                askpass_cmd.close()
            else:
                askpass_cmd = tempfile.NamedTemporaryFile(delete=False, suffix=".sh")
                askpass_cmd.write(b"#!/bin/sh\n")
                askpass_cmd.write(b"echo $SSH_ASKPASS_PASSWORD\n")
                askpass_cmd.close()
                os.chmod(askpass_cmd.name, 0o700)
            
            self._ssh_askpass_path = askpass_cmd.name
            _logger.info(f"Created SSH askpass script at {self._ssh_askpass_path}")
            env["SSH_ASKPASS"] = self._ssh_askpass_path
            env["SSH_ASKPASS_REQUIRE"] = "force"
            env["SSH_ASKPASS_PASSWORD"] = password
            if "DISPLAY" not in env:
                env["DISPLAY"] = "1"
        elif key_path and os.path.exists(key_path):
            args.extend(["-i", key_path])
            args.extend(["-o", "StrictHostKeyChecking=accept-new"])

        args.append(target)

        _logger.info(f"Starting SSH tunnel with command: {args}")

        try:
            self._ssh_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                env=env,
            )
            
            time.sleep(0.5)
            if self._ssh_process.poll() is not None:
                stderr = self._ssh_process.stderr.read().decode(errors="ignore")
                _logger.error(f"SSH Tunnel failed (return code {self._ssh_process.returncode}): {stderr}")
                self._stop_ssh_tunnel()
                return False
            
            return True
        except FileNotFoundError:
            _logger.error("SSH command not found. Please ensure 'ssh' is installed and in your PATH.")
            self._stop_ssh_tunnel()
            return False
        except Exception as e:
            _logger.error(f"Failed to start SSH tunnel: {e}")
            self._stop_ssh_tunnel()
            return False

    def _verify_remote_listening(self, credentials):
        """
        Diagnostics: Checks if the remote server is actually listening on the target port.
        """
        host = credentials.get("host")
        user = credentials.get("username")
        password = credentials.get("password")
        port = credentials.get("port", 22)
        remote_port = credentials.get("ssh_remote_port", 50051)
        remote_bind = credentials.get("ssh_remote_bind", "127.0.0.1")
        
        # Force 127.0.0.1 for the check to match the tunnel behavior
        if remote_bind == "localhost":
            remote_bind = "127.0.0.1"
        
        target = f"{user}@{host}"
        
        # Use PowerShell to check the port, as it's more reliable on Windows
        ps_command = f"$client = New-Object System.Net.Sockets.TcpClient; try {{ $client.Connect('{remote_bind}', {remote_port}); echo 'OPEN' }} catch {{ echo 'CLOSED' }} finally {{ $client.Dispose() }}"
        check_cmd = f"powershell -NoProfile -Command \"{ps_command}\""
        
        args = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=accept-new", target, check_cmd]
        
        env = os.environ.copy()
        if password:
            # Reuse the askpass setup from _start_ssh_tunnel logic (assuming env is set up there or here)
            # For simplicity, we assume the askpass script path is still valid if set
            if self._ssh_askpass_path:
                env["SSH_ASKPASS"] = self._ssh_askpass_path
                env["SSH_ASKPASS_REQUIRE"] = "force"
                env["SSH_ASKPASS_PASSWORD"] = password
                if "DISPLAY" not in env:
                    env["DISPLAY"] = "1"

        try:
            result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=10)
            status = result.stdout.strip()
            _logger.info(f"Remote port check output: '{status}'")
            if result.stderr:
                _logger.warning(f"Remote port check stderr: {result.stderr.strip()}")
            
            if status == "OPEN":
                _logger.info(f"Remote service is listening on {remote_bind}:{remote_port}")
            else:
                _logger.warning(f"Remote service is NOT listening on {remote_bind}:{remote_port} (status: {status})")
        except Exception as e:
            _logger.warning(f"Failed to verify remote port: {e}")

    def connect(self, credentials):
        """
        Establishes a connection to the remote server.
        """
        if "host" in credentials:
            self._config.hostname = credentials["host"]
        
        if self._channel:
            self._channel.close()
            self._channel = None
            
        self._stop_ssh_tunnel()
        
        target_address = self._config.address
        
        if self._config.hostname and self._config.hostname not in ["localhost", "127.0.0.1", ""]:
            _logger.info(f"Initializing SSH tunnel to {self._config.hostname}...")
            if self._start_ssh_tunnel(credentials):
                local_port = credentials.get("ssh_local_port", 50051)
                target_address = f"localhost:{local_port}"
            else:
                _logger.error("SSH tunnel initialization failed.")
                return False
            
        # Before creating the main channel, do a no-auth health check for diagnostics
        try:
            with grpc.insecure_channel(target_address) as temp_channel:
                temp_stub = pb2_grpc.HealthStub(temp_channel)
                resp = temp_stub.Check(pb2.HealthCheckRequest(), timeout=2)
                if resp.status == "SERVING":
                    _logger.info("No-auth health check passed.")
                else:
                    _logger.warning(f"No-auth health check returned status: {resp.status}")
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                _logger.warning("No-auth health check timed out. The remote server may be hung or unresponsive.")
            else:
                _logger.warning(f"No-auth health check failed: {e.details()}")
        except Exception as e:
            _logger.warning(f"No-auth health check failed with unexpected error: {e}")

        # Create channel with auth interceptor
        channel = grpc.insecure_channel(target_address)
        self._channel = grpc.intercept_channel(channel, AuthInterceptor(self._config.token))
        
        self._stub_health = pb2_grpc.HealthStub(self._channel)
        self._stub_file = pb2_grpc.FileServiceStub(self._channel)
        self._stub_inference = pb2_grpc.InferenceServiceStub(self._channel)
        
        # Retry health check to allow tunnel to establish
        for i in range(15):
            if self.health_check():
                _logger.info(f"Successfully connected to remote server at {target_address}")
                return True
            time.sleep(1)
            if self._ssh_process and self._ssh_process.poll() is not None:
                break

        _logger.error(f"Health check failed for remote server at {target_address}")
        
        # Run diagnostic check
        self._verify_remote_listening(credentials)
        
        if self._ssh_process:
            if self._ssh_process.poll() is None:
                _logger.info("SSH tunnel process is still running but health check failed.")
                self._ssh_process.terminate()
                try:
                    _, stderr = self._ssh_process.communicate(timeout=2)
                    if stderr:
                        _logger.error(f"SSH Tunnel stderr: {stderr.decode(errors='ignore')}")
                except Exception:
                    pass
            else:
                stderr = self._ssh_process.stderr.read().decode(errors='ignore')
                _logger.error(f"SSH Tunnel died with code {self._ssh_process.returncode}. stderr: {stderr}")
            self._stop_ssh_tunnel()
            
        return False

    def disconnect(self):
        """Closes the remote connection and any SSH tunnel."""
        if self._channel:
            try:
                self._channel.close()
            except Exception:
                pass
        self._channel = None
        self._stub_health = None
        self._stub_file = None
        self._stub_inference = None
        self._stop_ssh_tunnel()

    def download_file(self, uri, local_path):
        if not uri:
            raise ValueError("No URI provided")
        channel = grpc.insecure_channel(self._config.address)
        grpc_client.download_file(channel, uri, local_path, token=self._config.token)

    def health_check(self):
        if not self._stub_health:
            return False
        try:
            resp = self._stub_health.Check(pb2.HealthCheckRequest())
            return resp.status == "SERVING"
        except grpc.RpcError:
            return False

    def list_models(self):
        """
        Lists available models on the remote server.
        """
        if not self._stub_inference:
            return []
        try:
            resp = self._stub_inference.ListModels(pb2.ListModelsRequest())
            return list(resp.model_ids)
        except grpc.RpcError:
            return []

    def upload_file(self, project_id, file_path, progress_callback=None):
        if not self._stub_file:
            raise RuntimeError("Not connected")
            
        relpath = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        uploaded_bytes = 0
        
        def chunk_generator():
            nonlocal uploaded_bytes
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(1024 * 1024) # 1MB
                    if not data:
                        break
                    uploaded_bytes += len(data)
                    if progress_callback:
                        progress_callback(int(uploaded_bytes / file_size * 100))
                    yield pb2.FileChunk(
                        project_id=project_id,
                        relpath=relpath,
                        data=data
                    )
        
        response = self._stub_file.Upload(chunk_generator())
        return response.uri

    def upload_model(self, file_path, progress_callback=None):
        """Uploads a custom model to the server's model directory."""
        # The server treats uploads to _admin_models as model uploads
        return self.upload_file("_admin_models", file_path, progress_callback)

    def clear_user_jobs(self):
        """Clears the user's training jobs on the server."""
        if not self._stub_file:
            raise RuntimeError("Not connected")
        
        # The server recognizes this specific project/path combination as a clear command
        def chunk_generator():
            yield pb2.FileChunk(project_id="_admin", relpath="__clear_user_jobs__", data=b"")
            
        self._stub_file.Upload(chunk_generator())

    def download_file(self, uri, save_path):
        if not self._stub_file:
            raise RuntimeError("Not connected")
            
        response_iterator = self._stub_file.Download(pb2.DownloadRequest(uri=uri))
        
        with open(save_path, "wb") as f:
            for chunk in response_iterator:
                f.write(chunk.data)

    def run_inference(self, request):
        if not self._stub_inference:
            return iter([])
        return self._stub_inference.Run(request)
