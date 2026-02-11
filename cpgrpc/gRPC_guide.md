# gRPC Server and Client Guide

This document provides instructions on how to set up and run the gRPC server and client for Cellpose, details the available services and messages, and explains how to run the tests.

## 1. Setup

### Dependencies

Ensure you have the required dependencies installed. The main dependencies for the gRPC components are:

- `grpcio`
- `grpcio-tools`

You can install them using pip:

```bash
pip install grpcio grpcio-tools
```

### Running the Server

The gRPC server can be started by running the `services.py` script located in `cpgrpc/server`.

```bash
python -m cpgrpc.server.services
```

By default, the server starts on `localhost:50051`. You can bind it to a different address. The `serve` function in `services.py` takes a `bind` address string. You would need to modify the script to change the address.

The server requires a `CELLPOSE_SERVER_TOKEN` environment variable to be set for authentication.

```bash
export CELLPOSE_SERVER_TOKEN="your-secret-token"
python -m cpgrpc.server.services
```

### Running the Client

The gRPC client is implemented as a command-line interface (CLI) in `cpgrpc/client/client.py`.

You can see the available commands and arguments by running:

```bash
python -m cpgrpc.client.client --help
```

To run a health check:

```bash
python -m cpgrpc.client.client --address localhost:50051 health
```

To run an inference job, you need to provide a project ID, a model ID, and one or more image URIs. You also need to provide the authentication token.

```bash
python -m cpgrpc.client.client --address localhost:50051 --token "your-secret-token" run --project-id my_project --model-id my_model "file:///path/to/image1.png"
```

## 2. Services and Messages

The gRPC services and messages are defined in `cpgrpc/protos/cellpose_remote.proto`.

### Services

- **`Health`**: A simple service to check the health of the server.
  - `Check(HealthCheckRequest) returns (HealthCheckResponse)`: Returns the status of the server.

- **`FileService`**: A service to upload files to the server.
  - `Upload(stream FileChunk) returns (UploadReply)`: Uploads a file in chunks.

- **`InferenceService`**: A service to run inference on images.
  - `Run(RunRequest) returns (stream JobUpdate)`: Starts an inference job and streams updates on its progress.

### Messages

- **`HealthCheckRequest`**: Empty message for the health check.
- **`HealthCheckResponse`**: Contains the server status.
  - `status` (string): e.g., "SERVING".

- **`FileChunk`**: A chunk of a file being uploaded.
  - `project_id` (string): The project the file belongs to.
  - `relpath` (string): The relative path of the file within the project.
  - `data` (bytes): The file content.
  - `offset` (int64): The byte offset for resumable uploads.

- **`UploadReply`**: The reply after a file upload.
  - `uri` (string): The server URI for the stored file.
  - `bytes_received` (int64): The total number of bytes received.

- **`RunRequest`**: A request to run an inference job.
  - `project_id` (string): The project ID.
  - `uris` (repeated string): The URIs of the images to process.
  - `model_id` (string): The model to use for inference.

- **`JobUpdate`**: An update on the progress of an inference job.
  - `progress` (int32): The progress of the job (0-100).
  - `stage` (string): The current stage of the job (e.g., "queue", "preprocess", "infer", "postprocess").
  - `message` (string): A free-form log message.
  - `result_uri` (string): An optional URI to the output.

## 3. Testing

The tests are written using `pytest`. You can run the tests to confirm that all aspects of the gRPC functionality are working correctly.

First, install `pytest`:

```bash
pip install pytest
```

Then, run the tests from the root of the project:

```bash
python -m pytest
```

This will discover and run all the tests, including the gRPC tests. The gRPC tests start a real server on a free port and run client requests against it.

To run only the gRPC client CLI tests:

```bash
python -m pytest tests/test_grpc_client_cli.py
```
