import logging

from cellpose import models, core
import os
import numpy as np
import copy

_logger = logging.getLogger(__name__)


_LEGACY_EVAL_DEFAULTS = {
    "cellprob_threshold": -0.5,
    "flow_threshold": 1.0,
    "do_3D": False,
    "niter": 0,
    "stitch_threshold": 0.0,
    "anisotropy": 1.0,
    "flow3D_smooth": 0.0,
    "min_size": 15,
    "max_size_fraction": 1.0,
}


class SegmentationService:
    """
    Service for running local segmentation inference using Cellpose.
    """
    def __init__(self):
        self.current_model = None
        self.current_model_id = None
        self.use_gpu = core.use_gpu()

    def load_model(self, model_id):
        """Loads the model if it's not already loaded."""
        if self.current_model_id == model_id and self.current_model is not None:
            return self.current_model
            
        # Use pretrained_model for both core and custom models (model_type is ignored in v4.0.1+)
        pretrained_model = model_id
        if model_id:
            if os.path.exists(model_id):
                pretrained_model = model_id
            else:
                try:
                    candidate = models.MODEL_DIR.joinpath(model_id)
                    if candidate.exists():
                        pretrained_model = os.fspath(candidate)
                except Exception:
                    pretrained_model = model_id
        self.current_model = models.CellposeModel(gpu=self.use_gpu, pretrained_model=pretrained_model)
        _logger.info("Resolved pretrained_model path: %s", getattr(self.current_model, "pretrained_model", pretrained_model))
            
        self.current_model_id = model_id
        return self.current_model

    def run_inference(self, image, diameter, model_id="cpsam", channel_index=None):
        """Runs inference on the provided image."""
        model = self.load_model(model_id)
        if channel_index is not None and image is not None:
            try:
                if image.ndim == 3:
                    # Respect user channel selection for both channels-last (H, W, C)
                    # and channels-first (C, H, W) layouts.
                    if image.shape[-1] > channel_index and image.shape[-1] <= 4:
                        image = image[..., channel_index]
                    elif image.shape[0] > channel_index and image.shape[0] <= 4:
                        image = image[channel_index, ...]
                    elif image.shape[2] > channel_index:
                        image = image[..., channel_index]
                    else:
                        _logger.warning(
                            "Requested channel_index=%s is out of range for image shape %s; using original image.",
                            channel_index,
                            getattr(image, "shape", None),
                        )
            except Exception:
                pass

        normalize_params = copy.deepcopy(models.normalize_default)
        normalize_params["normalize"] = True

        channel_axis = None
        if image is not None and hasattr(image, "ndim") and image.ndim == 3:
            if image.shape[-1] <= 4:
                channel_axis = -1
            elif image.shape[0] <= 4:
                channel_axis = 0

        eval_kwargs = dict(_LEGACY_EVAL_DEFAULTS)
        eval_kwargs.update(
            {
                "diameter": diameter,
                "normalize": normalize_params,
                "channel_axis": channel_axis,
                "z_axis": None,
                "progress": None,
            }
        )

        # eval returns masks, flows, styles
        masks, flows, styles = model.eval(image, **eval_kwargs)
        return masks, flows, styles

    def postprocess_classes(self, masks, styles):
        """Extracts class information from model styles/logits if available."""
        classes = None
        classes_map = None
        
        if styles is not None:
            arr = np.squeeze(styles)
            if arr.ndim >= 3 and arr.shape[-1] > 1:
                classes_map = np.argmax(arr, axis=-1).astype(np.int32)
                
        if classes_map is not None and masks is not None:
            masks_arr = masks
            classes_arr = classes_map
            if masks_arr.ndim == 3 and masks_arr.shape[0] == 1:
                masks_arr = masks_arr[0]
            if classes_arr.ndim == 3 and classes_arr.shape[0] == 1:
                classes_arr = classes_arr[0]
            if masks_arr.shape != classes_arr.shape:
                _logger.warning(
                    "Class map shape %s does not match masks %s. Resizing class map.",
                    classes_arr.shape,
                    masks_arr.shape,
                )
                try:
                    import cv2
                    classes_arr = cv2.resize(
                        classes_arr.astype(np.int32),
                        (masks_arr.shape[1], masks_arr.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                except Exception:
                    try:
                        from skimage.transform import resize as sk_resize
                        classes_arr = sk_resize(
                            classes_arr,
                            masks_arr.shape,
                            order=0,
                            preserve_range=True,
                            anti_aliasing=False,
                        ).astype(np.int32)
                    except Exception as exc:
                        _logger.warning(
                            "Failed to resize class map to mask shape: %s. Skipping class postprocess.",
                            exc,
                        )
                        return None, None
            nmask = int(masks_arr.max())
            classes = np.zeros(nmask + 1, dtype=np.int16)
            if nmask > 0:
                for mid in range(1, nmask + 1):
                    vals = classes_arr[masks_arr == mid]
                    vals = vals[vals > 0]
                    if vals.size > 0:
                        counts = np.bincount(vals.astype(np.int64))
                        classes[mid] = int(np.argmax(counts))
                        
        return classes, classes_map
