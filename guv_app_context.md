# guv_app code context

Short, one-sentence descriptions of every **top-level** class/function (and class methods) in the `guv_app` package.

## `__init__.py`
- *(No top-level classes or functions in this module.)*

## `controllers/__init__.py`
- *(No top-level classes or functions in this module.)*

## `controllers/main_controller.py`
- **class `MainController`** — The Main Controller for the application.
  - **`__init__(self, model, view, services)`** — Initialize the controller and store references to the MVC components.
  - **`connect_signals(self)`** — Connect UI signals from the View to the controller's handler methods.
  - **`handle_run_inference(self)`** — Handle the "Run Inference" event.
  - **`handle_load_image(self)`** — Handle the "Load Image" event.

## `data_models/__init__.py`
- *(No top-level classes or functions in this module.)*

## `data_models/configs.py`
- **class `InferenceConfig`** — Bundles all parameters needed to run segmentation.
  - *(No methods defined in this class.)*
- **class `TrainingConfig`** — Bundles all parameters for training a new model.
  - *(No methods defined in this class.)*
- **class `ViewConfig`** — Holds the state of the UI's view controls.
  - *(No methods defined in this class.)*
- **class `RemoteConfig`** — Contains all settings for connecting to a remote server.
  - *(No methods defined in this class.)*

## `data_models/entities.py`
- **class `ImageFile`** — Represents an image loaded into the application.
  - *(No methods defined in this class.)*
- **class `SegmentationMask`** — Represents a single segmented mask/ROI as an object.
  - *(No methods defined in this class.)*

## `data_models/results.py`
- **class `InferenceResult`** — A standard container for the output of a segmentation run.
  - *(No methods defined in this class.)*
- **class `TrainingResult`** — A standard container for the output of a training run.
  - *(No methods defined in this class.)*

## `main.py`
- **`main()`** — Main function to initialize and run the GUVpose application.

## `models/__init__.py`
- *(No top-level classes or functions in this module.)*

## `models/app_state.py`
- **class `ApplicationStateModel`** — A central, observable class holding the application's shared state.
  - **`__init__(self)`** — Initializer that sets up instance state.

## `services/__init__.py`
- *(No top-level classes or functions in this module.)*

## `services/image_service.py`
- **class `ImageService`** — Manages all file I/O and data conversion related to images.
  - **`__init__(self)`** — Initializer that sets up instance state.
  - **`load_image(self, path)`** — Loads data from disk or an external source.

## `services/model_management_service.py`
- **class `ModelManagementService`** — Manages the available segmentation models.
  - **`__init__(self)`** — Initializer that sets up instance state.
  - **`get_local_models(self)`** — Returns requested data or computed value.

## `services/remote_service.py`
- **class `RemoteConnectionService`** — Manages the state and logic for remote gRPC connections.
  - **`__init__(self)`** — Initializer that sets up instance state.
  - **`connect(self, hostname, credentials)`** — Function definition.

## `services/segmentation_service.py`
- **class `SegmentationService`** — Manages the core segmentation and inference logic.
  - **`__init__(self)`** — Initializer that sets up instance state.
  - **`run_inference(self, image, model, params)`** — Entry-point routine that orchestrates execution.

## `services/training_service.py`
- **class `TrainingService`** — Manages the model training pipeline.
  - **`__init__(self)`** — Initializer that sets up instance state.
  - **`start_training(self, params)`** — Function definition.

## `views/__init__.py`
- *(No top-level classes or functions in this module.)*

## `views/analyzer_view.py`
- **class `AnalyzerView`(BaseMainView)** — The view for the Analyzer, inheriting from the BaseMainView.
  - **`__init__(self, parent)`** — Initializer that sets up instance state.

## `views/base_view.py`
- **class `BaseMainView`(QMainWindow)** — The main application window containing all shared UI components.
  - **`__init__(self, parent)`** — Initializer that sets up instance state.
  - **`display_image(self, image_data)`** — Method to render image data.
  - **`draw_masks(self, mask_data)`** — Method to render masks.
  - **`update_model_list(self, models)`** — Method to populate the model dropdown.

## `views/dialogs/__init__.py`
- *(No top-level classes or functions in this module.)*

## `views/trainer_view.py`
- **class `TrainerView`(BaseMainView)** — The view for the Trainer, inheriting from the BaseMainView.
  - **`__init__(self, parent)`** — Initializer that sets up instance state.

## `views/widgets/__init__.py`
- *(No top-level classes or functions in this module.)*

## `workers/__init__.py`
- *(No top-level classes or functions in this module.)*

## `workers/analysis_worker.py`
- **class `AnalysisWorker`(BaseWorker)** — Worker for running analysis on a folder of images.
  - **`run(self)`** — Entry-point routine that orchestrates execution.

## `workers/base_worker.py`
- **class `BaseWorker`(QObject)** — A base worker class that inherits from QObject to leverage Qt's
  - **`__init__(self, *args, **kwargs)`** — Initializer that sets up instance state.
  - **`run(self)`** — The main work method.

## `workers/file_download_worker.py`
- **class `FileDownloadWorker`(BaseWorker)** — Worker for downloading files from a remote server.
  - **`run(self)`** — Entry-point routine that orchestrates execution.

## `workers/file_upload_worker.py`
- **class `FileUploadWorker`(BaseWorker)** — Worker for uploading files to a remote server.
  - **`run(self)`** — Entry-point routine that orchestrates execution.

## `workers/inference_worker.py`
- **class `InferenceWorker`(BaseWorker)** — Worker for running segmentation inference on the local machine.
  - **`run(self)`** — Entry-point routine that orchestrates execution.

## `workers/mask_load_worker.py`
- **class `MaskLoadWorker`(BaseWorker)** — Worker for loading and processing masks from disk.
  - **`run(self)`** — Entry-point routine that orchestrates execution.

## `workers/remote_inference_worker.py`
- **class `RemoteInferenceWorker`(BaseWorker)** — Worker for running segmentation inference on a remote server.
  - **`run(self)`** — Entry-point routine that orchestrates execution.

## `workers/remote_training_worker.py`
- **class `RemoteTrainingWorker`(BaseWorker)** — Worker for running model training on a remote server.
  - **`run(self)`** — Entry-point routine that orchestrates execution.

## `workers/training_worker.py`
- **class `TrainingWorker`(BaseWorker)** — Worker for running model training on the local machine.
  - **`run(self)`** — Entry-point routine that orchestrates execution.
