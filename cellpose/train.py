import time
import os
import numpy as np
from cellpose import io, utils, models, dynamics
from cellpose.transforms import normalize_img, random_rotate_and_resize
from pathlib import Path
import torch
from torch import nn
from tqdm import trange

import logging

train_logger = logging.getLogger(__name__)

_TRAIN_DEBUG_ENABLED = False
_TRAIN_DEBUG_STEPS = 3


def set_train_debug(enabled=False, steps=3):
    """Enable/disable temporary train-loop debug logging."""
    global _TRAIN_DEBUG_ENABLED, _TRAIN_DEBUG_STEPS
    _TRAIN_DEBUG_ENABLED = bool(enabled)
    try:
        _TRAIN_DEBUG_STEPS = max(1, int(steps))
    except Exception:
        _TRAIN_DEBUG_STEPS = 3


def _cuda_mem_stats(device):
    if not torch.cuda.is_available():
        return "cuda=n/a"
    if device is None or getattr(device, "type", None) != "cuda":
        return "cuda=not-active"
    idx = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(idx) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated(idx) / (1024 ** 2)
    return f"alloc={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB"

def _loss_fn_class(lbl, y, class_weights=None):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        
    Returns:
        torch.Tensor: Loss value.

    """

    # ensure class_weights length matches number of class logits (exclude last 3 seg channels)
    ncls = y.shape[1] - 3 if y.ndim >= 2 else 0
    if class_weights is not None:
        try:
            # if provided as tensor/array, check first dimension
            if hasattr(class_weights, 'shape'):
                if class_weights.shape[0] != ncls:
                    class_weights = None
        except Exception:
            class_weights = None
    # number of class channels predicted (exclude last 3 seg channels)
    ncls = y.shape[1] - 3 if y.ndim >= 2 else 0
    
    if ncls <= 0:
        return 0. * y.sum()

    # choose class-target channel from lbl
    # convention with GUI semantic labels: channel 1 stores per-pixel class ids
    if lbl.ndim >= 3 and lbl.shape[1] > 1:
        # print(f"taking class dimension as 1st dimension")
        tgt = torch.round(lbl[:, 1]).long()
        # print(f" after rounding unique class ids are {torch.unique(tgt)}")
    else:
        # fallback: use first channel but clamp to valid range
        tgt = torch.round(lbl[:, 0]).long() if lbl.ndim >= 3 else torch.zeros_like(y[:, -1], dtype=torch.long)
    # clamp targets to [0, ncls-1] to avoid CUDA assert
    tgt = tgt.clamp(min=0, max=max(0, ncls - 1))
    # print(f">> after clamping target values are {torch.unique(tgt)}")
    # print(f">> target shape: {tgt.shape}")
    # print(f">> network class logits shape: {y[:,:-3].shape}")
    # validate/adjust class weights
    if class_weights is not None:
        try:
            if hasattr(class_weights, 'shape'):
                if class_weights.shape[0] != ncls:
                    class_weights = None
        except Exception:
            class_weights = None
    criterion3 = nn.CrossEntropyLoss(reduction="mean", weight=class_weights)
    loss3 = criterion3(y[:, :-3], tgt)
    
    return loss3

def _loss_fn_seg(lbl, y, device):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        device (torch.device): Device on which the tensors are located.

    Returns:
        torch.Tensor: Loss value.

    """
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    # segmentation loss is defined on the last 3 outputs of y: [flowY, flowX, cellprob]
    # be robust to varying label channel layouts
    # flows target (2 channels)
    if lbl.shape[1] >= 2:
        veci = 5. * lbl[:, -2:]
    else:
        veci = torch.zeros_like(y[:, -3:-1])
    loss = criterion(y[:, -3:-1], veci)
    loss /= 2.
    # cellprob target (1 channel)
    if lbl.shape[1] >= 3:
        cp_lbl = (lbl[:, -3] > 0.5).float()
    else:
        cp_lbl = (lbl[:, 0] > 0.5).float()
    loss2 = criterion2(y[:, -1], cp_lbl)
    loss = loss + loss2
    return loss

def _reshape_norm(data, channel_axis=None, normalize_params={"normalize": False}):
    """
    Reshapes and normalizes the input data.

    Args:
        data (list): List of input data, with channels axis first or last.
        normalize_params (dict, optional): Dictionary of normalization parameters. Defaults to {"normalize": False}.

    Returns:
        list: List of reshaped and normalized data.
    """
    if (np.array([td.ndim!=3 for td in data]).sum() > 0 or
        np.array([td.shape[0]!=3 for td in data]).sum() > 0):
        data_new = []
        for td in data:
            if td.ndim == 3:
                channel_axis0 = channel_axis if channel_axis is not None else np.array(td.shape).argmin()
                # put channel axis first 
                td = np.moveaxis(td, channel_axis0, 0)
                td = td[:3] # keep at most 3 channels
            if td.ndim == 2 or (td.ndim == 3 and td.shape[0] == 1):
                td = np.stack((td, 0*td, 0*td), axis=0)
            elif td.ndim == 3 and td.shape[0] < 3:
                td = np.concatenate((td, 0*td[:1]), axis=0)
            data_new.append(td)
        data = data_new
    if normalize_params["normalize"]:
        data = [
            normalize_img(td, normalize=normalize_params, axis=0)
            for td in data
        ]
    return data

def _get_batch(inds, data=None, labels=None, files=None, labels_files=None,
               normalize_params={"normalize": False}):
    """
    Get a batch of images and labels.

    Args:
        inds (list): List of indices indicating which images and labels to retrieve.
        data (list or None): List of image data. If None, images will be loaded from files.
        labels (list or None): List of label data. If None, labels will be loaded from files.
        files (list or None): List of file paths for images.
        labels_files (list or None): List of file paths for labels.
        normalize_params (dict): Dictionary of parameters for image normalization (will be faster, if loading from files to pre-normalize).

    Returns:
        tuple: A tuple containing two lists: the batch of images and the batch of labels.
    """
    if data is None:
        lbls = None
        imgs = [io.imread(files[i]) for i in inds]
        imgs = _reshape_norm(imgs, normalize_params=normalize_params)
        if labels_files is not None:
            lbls = [io.imread(labels_files[i])[1:] for i in inds]
    else:
        imgs = [data[i] for i in inds]
        lbls = [labels[i] for i in inds]
    return imgs, lbls

def _reshape_norm_save(files, channels=None, channel_axis=None,
                       normalize_params={"normalize": False}):
    """ not currently used -- normalization happening on each batch if not load_files """
    files_new = []
    for f in trange(files):
        td = io.imread(f)
        if channels is not None:
            td = convert_image(td, channels=channels,
                                          channel_axis=channel_axis)
            td = td.transpose(2, 0, 1)
        if normalize_params["normalize"]:
            td = normalize_img(td, normalize=normalize_params, axis=0)
        fnew = os.path.splitext(str(f))[0] + "_cpnorm.tif"
        io.imsave(fnew, td)
        files_new.append(fnew)
    return files_new
    # else:
    #     train_files = reshape_norm_save(train_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)
    # elif test_files is not None:
    #     test_files = reshape_norm_save(test_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)


def _process_train_test(train_data=None, train_labels=None, train_files=None,
                        train_labels_files=None, train_probs=None, test_data=None,
                        test_labels=None, test_files=None, test_labels_files=None,
                        test_probs=None, load_files=True, min_train_masks=5,
                        compute_flows=False, normalize_params={"normalize": False}, 
                        channel_axis=None, device=None):
    """
    Process train and test data.

    Args:
        train_data (list or None): List of training data arrays.
        train_labels (list or None): List of training label arrays.
        train_files (list or None): List of training file paths.
        train_labels_files (list or None): List of training label file paths.
        train_probs (ndarray or None): Array of training probabilities.
        test_data (list or None): List of test data arrays.
        test_labels (list or None): List of test label arrays.
        test_files (list or None): List of test file paths.
        test_labels_files (list or None): List of test label file paths.
        test_probs (ndarray or None): Array of test probabilities.
        load_files (bool): Whether to load data from files.
        min_train_masks (int): Minimum number of masks required for training images.
        compute_flows (bool): Whether to compute flows.
        channels (list or None): List of channel indices to use.
        channel_axis (int or None): Axis of channel dimension.
        rgb (bool): Convert training/testing images to RGB.
        normalize_params (dict): Dictionary of normalization parameters.
        device (torch.device): Device to use for computation.

    Returns:
        tuple: A tuple containing the processed train and test data and sampling probabilities and diameters.
    """
    if device == None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else None
    
    if train_data is not None and train_labels is not None:
        # if data is loaded
        nimg = len(train_data)
        nimg_test = len(test_data) if test_data is not None else None
    else:
        # otherwise use files
        nimg = len(train_files)
        if train_labels_files is None:
            train_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in train_files
            ]
            train_labels_files = [tf for tf in train_labels_files if os.path.exists(tf)]
        if (test_data is not None or
                test_files is not None) and test_labels_files is None:
            test_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in test_files
            ]
            test_labels_files = [tf for tf in test_labels_files if os.path.exists(tf)]
        if not load_files:
            train_logger.info(">>> using files instead of loading dataset")
        else:
            # load all images
            train_logger.info(">>> loading images and labels")
            train_data = [io.imread(train_files[i]) for i in trange(nimg)]
            train_labels = [io.imread(train_labels_files[i]) for i in trange(nimg)]
        nimg_test = len(test_files) if test_files is not None else None
        if load_files and nimg_test:
            test_data = [io.imread(test_files[i]) for i in trange(nimg_test)]
            test_labels = [io.imread(test_labels_files[i]) for i in trange(nimg_test)]

    ### check that arrays are correct size
    if ((train_labels is not None and nimg != len(train_labels)) or
        (train_labels_files is not None and nimg != len(train_labels_files))):
        error_message = "train data and labels not same length"
        train_logger.critical(error_message)
        raise ValueError(error_message)
    if ((test_labels is not None and nimg_test != len(test_labels)) or
        (test_labels_files is not None and nimg_test != len(test_labels_files))):
        train_logger.warning("test data and labels not same length, not using")
        test_data, test_files = None, None
    if train_labels is not None:
        if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
            error_message = "training data or labels are not at least two-dimensional"
            train_logger.critical(error_message)
            raise ValueError(error_message)
        if train_data[0].ndim > 3:
            error_message = "training data is more than three-dimensional (should be 2D or 3D array)"
            train_logger.critical(error_message)
            raise ValueError(error_message)

    ### check that flows are computed
    if train_labels is not None:
        # print(f"Flows are computed, converting to flow ares")
        train_labels = dynamics.labels_to_flows(train_labels, files=train_files,
                                                device=device)
        
        # print(f"train label shape: {train_labels[0].shape}")
        
        if test_labels is not None:
            test_labels = dynamics.labels_to_flows(test_labels, files=test_files,
                                                   device=device)
    elif compute_flows:
        for k in trange(nimg):
            tl = dynamics.labels_to_flows(io.imread(train_labels_files),
                                          files=train_files, device=device)
        if test_files is not None:
            for k in trange(nimg_test):
                tl = dynamics.labels_to_flows(io.imread(test_labels_files),
                                              files=test_files, device=device)

    ### compute diameters
    nmasks = np.zeros(nimg)
    diam_train = np.zeros(nimg)
    train_logger.info(">>> computing diameters")
    for k in trange(nimg):
        tl = (train_labels[k][0]
              if train_labels is not None else io.imread(train_labels_files[k])[0])
        diam_train[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam_train[diam_train < 5] = 5.
    if test_data is not None:
        diam_test = np.array(
            [utils.diameters(test_labels[k][0])[0] for k in trange(len(test_labels))])
        diam_test[diam_test < 5] = 5.
    elif test_labels_files is not None:
        diam_test = np.array([
            utils.diameters(io.imread(test_labels_files[k])[0])[0]
            for k in trange(len(test_labels_files))
        ])
        diam_test[diam_test < 5] = 5.
    else:
        diam_test = None

    ### check to remove training images with too few masks
    if min_train_masks > 0:
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            train_logger.warning(
                f"{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set"
            )
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            if train_data is not None:
                train_data = [train_data[i] for i in ikeep]
                train_labels = [train_labels[i] for i in ikeep]
            if train_files is not None:
                train_files = [train_files[i] for i in ikeep]
            if train_labels_files is not None:
                train_labels_files = [train_labels_files[i] for i in ikeep]
            if train_probs is not None:
                train_probs = train_probs[ikeep]
            diam_train = diam_train[ikeep]
            nimg = len(train_data)

    ### normalize probabilities
    train_probs = 1. / nimg * np.ones(nimg,
                                      "float64") if train_probs is None else train_probs
    train_probs /= train_probs.sum()
    if test_files is not None or test_data is not None:
        test_probs = 1. / nimg_test * np.ones(
            nimg_test, "float64") if test_probs is None else test_probs
        test_probs /= test_probs.sum()

    ### reshape and normalize train / test data
    normed = False
    if normalize_params["normalize"]:
        train_logger.info(f">>> normalizing {normalize_params}")
    if train_data is not None:
        train_data = _reshape_norm(train_data, channel_axis=channel_axis, 
                                   normalize_params=normalize_params)
        normed = True
    if test_data is not None:
        test_data = _reshape_norm(test_data, channel_axis=channel_axis,
                                  normalize_params=normalize_params)

    return (train_data, train_labels, train_files, train_labels_files, train_probs,
            diam_train, test_data, test_labels, test_files, test_labels_files,
            test_probs, diam_test, normed)


def train_seg(net, train_data=None, train_labels=None, train_files=None,
              train_labels_files=None, train_probs=None, test_data=None,
              test_labels=None, test_files=None, test_labels_files=None,
              test_probs=None, channel_axis=None,
              load_files=True, batch_size=1, learning_rate=5e-5, SGD=False,
              n_epochs=100, weight_decay=0.1, normalize=True, compute_flows=False,
              save_path=None, save_every=100, save_each=False, nimg_per_epoch=None,
              nimg_test_per_epoch=None, rescale=False, scale_range=None, bsize=256,
              min_train_masks=5, model_name=None, class_weights=None, seg_loss_weight = 0.1,
              early_stop=False, patience=3, min_delta=0.0, progress_callback=None):
    """
    Train the network with images for segmentation.

    Args:
        net (object): The network model to train.
        train_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for training. Defaults to None.
        train_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for train_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        train_files (List[str], optional): List of strings - file names for images in train_data (to save flows for future runs). Defaults to None.
        train_labels_files (list or None): List of training label file paths. Defaults to None.
        train_probs (List[float], optional): List of floats - probabilities for each image to be selected during training. Defaults to None.
        test_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for testing. Defaults to None.
        test_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for test_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        test_files (List[str], optional): List of strings - file names for images in test_data (to save flows for future runs). Defaults to None.
        test_labels_files (list or None): List of test label file paths. Defaults to None.
        test_probs (List[float], optional): List of floats - probabilities for each image to be selected during testing. Defaults to None.
        load_files (bool, optional): Boolean - whether to load images and labels from files. Defaults to True.
        batch_size (int, optional): Integer - number of patches to run simultaneously on the GPU. Defaults to 8.
        learning_rate (float or List[float], optional): Float or list/np.ndarray - learning rate for training. Defaults to 0.005.
        n_epochs (int, optional): Integer - number of times to go through the whole training set during training. Defaults to 2000.
        weight_decay (float, optional): Float - weight decay for the optimizer. Defaults to 1e-5.
        momentum (float, optional): Float - momentum for the optimizer. Defaults to 0.9.
        SGD (bool, optional): Deprecated in v4.0.1+ - AdamW always used.
        normalize (bool or dict, optional): Boolean or dictionary - whether to normalize the data. Defaults to True.
        compute_flows (bool, optional): Boolean - whether to compute flows during training. Defaults to False.
        save_path (str, optional): String - where to save the trained model. Defaults to None.
        save_every (int, optional): Integer - save the network every [save_every] epochs. Defaults to 100.
        save_each (bool, optional): Boolean - save the network to a new filename at every [save_each] epoch. Defaults to False.
        nimg_per_epoch (int, optional): Integer - minimum number of images to train on per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): Integer - minimum number of images to test on per epoch. Defaults to None.
        rescale (bool, optional): Boolean - whether or not to rescale images during training. Defaults to True.
        min_train_masks (int, optional): Integer - minimum number of masks an image must have to use in the training set. Defaults to 5.
        model_name (str, optional): String - name of the network. Defaults to None.

    Returns:
        tuple: A tuple containing the path to the saved model weights, training losses, and test losses.
       
    """
    if SGD:
        train_logger.warning("SGD is deprecated, using AdamW instead")

    device = net.device

    if not hasattr(net, "diam_mean") or net.diam_mean is None or not torch.is_tensor(net.diam_mean):
        train_logger.warning("train_seg: net.diam_mean missing; defaulting to 30.0")
        net.diam_mean = nn.Parameter(torch.tensor([30.0]), requires_grad=False)
    if not hasattr(net, "diam_labels") or net.diam_labels is None or not torch.is_tensor(net.diam_labels):
        train_logger.warning("train_seg: net.diam_labels missing; defaulting to 30.0")
        net.diam_labels = nn.Parameter(torch.tensor([30.0]), requires_grad=False)
    try:
        diam_mean_value = float(net.diam_mean.item())
    except Exception:
        diam_mean_value = 30.0
        try:
            net.diam_mean = nn.Parameter(torch.tensor([diam_mean_value]), requires_grad=False)
        except Exception:
            pass

    scale_range = 0.5 if scale_range is None else scale_range

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    out = _process_train_test(train_data=train_data, train_labels=train_labels,
                              train_files=train_files, train_labels_files=train_labels_files,
                              train_probs=train_probs,
                              test_data=test_data, test_labels=test_labels,
                              test_files=test_files, test_labels_files=test_labels_files,
                              test_probs=test_probs,
                              load_files=load_files, min_train_masks=min_train_masks,
                              compute_flows=compute_flows, channel_axis=channel_axis,
                              normalize_params=normalize_params, device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
    try:
        train_logger.info(
            "train_seg: diam_mean=%s diam_labels=%s rescale=%s scale_range=%.3f nimg=%s nimg_test=%s",
            diam_mean_value,
            getattr(net.diam_labels, "item", lambda: net.diam_labels)(),
            bool(rescale),
            float(scale_range),
            len(train_data) if train_data is not None else len(train_files),
            len(test_data) if test_data is not None else len(test_files) if test_files is not None else 0,
        )
    except Exception:
        pass
    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {"normalize_params": normalize_params, "channel_axis": channel_axis}
    
    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    if class_weights is not None and isinstance(class_weights, (list, np.ndarray, tuple)):
        class_weights = torch.from_numpy(class_weights).to(device).float()
        # print(class_weights)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    # learning rate schedule
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 99:
        LR = LR[:-50]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    print(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")
    print(
        f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}"
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay)

    debug_train = _TRAIN_DEBUG_ENABLED
    debug_steps = _TRAIN_DEBUG_STEPS
    if debug_train:
        try:
            total_params = 0
            trainable_params = 0
            lora_trainable = 0
            out_trainable = 0
            unexpected_encoder = []
            has_lora = False
            for pname, p in net.named_parameters():
                n = int(p.numel())
                total_params += n
                if not p.requires_grad:
                    continue
                trainable_params += n
                is_lora = (".lora_A" in pname) or (".lora_B" in pname)
                if is_lora:
                    has_lora = True
                if is_lora:
                    lora_trainable += n
                if pname.startswith("out."):
                    out_trainable += n
                if pname.startswith("encoder.") and not is_lora:
                    unexpected_encoder.append(pname)
            train_logger.info(
                "TRAIN_DEBUG: model params total=%s trainable=%s lora_trainable=%s out_trainable=%s",
                total_params,
                trainable_params,
                lora_trainable,
                out_trainable,
            )
            if has_lora and unexpected_encoder:
                train_logger.warning(
                    "TRAIN_DEBUG: unexpected trainable encoder params in LoRA mode (n=%s): %s",
                    len(unexpected_encoder),
                    unexpected_encoder[:20],
                )
        except Exception as exc:
            train_logger.warning("TRAIN_DEBUG: failed to compute trainability summary (%s)", exc)

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    # resolve save directory
    if save_path is None:
        save_dir = Path.cwd() / "models"
    else:
        save_dir = Path(save_path)
    # if save_dir is a file, back off to its parent
    if save_dir.exists() and not save_dir.is_dir():
        save_dir = save_dir.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / model_name

    train_logger.info(f">>> saving model to {filename}")

    train_logger.info(f">>> running on device: {net.device}")
    train_logger.info(f">>> images per epoch: {nimg_per_epoch}")

    if progress_callback is not None:
        try:
            progress_callback(0, "prep", "Preparing training data")
        except Exception:
            pass
    
    lavg, lavgc,nsum = 0, 0,0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    best_val = np.inf
    es_counter = 0
    for iepoch in range(n_epochs):
        if debug_train and torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
            try:
                idx = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
                torch.cuda.reset_peak_memory_stats(idx)
            except Exception:
                pass
        test_loss_epoch = None
        test_class_loss_epoch = None
        if progress_callback is not None:
            try:
                pct = int((iepoch / max(1, n_epochs)) * 100)
                progress_callback(pct, "train", f"Epoch {iepoch + 1}/{n_epochs}")
            except Exception:
                pass
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            # choose random images for epoch with probability train_probs
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            # otherwise use all images
            rperm = np.random.permutation(np.arange(0, nimg))
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch] # set learning rate
            
        train_logger.info(f">>> about to train ...")
        net.train()
        debug_train_logged = 0
        for k in range(0, nimg_per_epoch, batch_size):
            t0_batch = time.perf_counter()
            kend = min(k + batch_size, nimg_per_epoch)
            inds = rperm[k:kend]
            t0_get = time.perf_counter()
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels,
                                    files=train_files, labels_files=train_labels_files,
                                    **kwargs)
            t_get = time.perf_counter() - t0_get
            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / diam_mean_value if rescale else np.ones(
                len(diams), "float32")
            # augmentations
            t0_aug = time.perf_counter()
            imgi, lbl = random_rotate_and_resize(imgs, Y=lbls, rescale=rsc,
                                                            scale_range=scale_range,
                                                            xy=(bsize, bsize))[:2]
            t_aug = time.perf_counter() - t0_aug
            
            # print(f"After augmentations: {lbl[0].shape}")
            # print(f"After augmentation labels shape:{lbl[0].shape} ")
            # network and loss optimization
            t0_to = time.perf_counter()
            X = torch.from_numpy(imgi).to(device)
            lbl = torch.from_numpy(lbl).to(device)
            t_to = time.perf_counter() - t0_to
            t0_fwd = time.perf_counter()
            y = net(X)[0]
            loss = _loss_fn_seg(lbl, y, device)
            loss3 = None
            if y.shape[1] > 3:
                # train_logger.info(f">>> calculating class loss ...")
                loss3 = _loss_fn_class(lbl, y, class_weights=class_weights)
                loss = loss + loss3
            t_fwd = time.perf_counter() - t0_fwd
            t0_bwd = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_bwd = time.perf_counter() - t0_bwd
            train_loss = loss.item()
            class_loss = loss3.item() if loss3 is not None else 0.0
            train_loss *= len(imgi)
            class_loss *= len(imgi)
            # keep track of average training loss across epochs
            lavg += train_loss
            lavgc += class_loss
            nsum += len(imgi)
            # per epoch training loss
            train_losses[iepoch] += train_loss
            if debug_train and debug_train_logged < debug_steps:
                t_batch = time.perf_counter() - t0_batch
                train_logger.info(
                    "TRAIN_DEBUG: train epoch=%s batch=%s size=%s xshape=%s yshape=%s "
                    "t_get=%.3fs t_aug=%.3fs t_to=%.3fs t_fwd=%.3fs t_bwd=%.3fs t_total=%.3fs %s",
                    iepoch + 1,
                    (k // batch_size) + 1,
                    len(inds),
                    tuple(X.shape),
                    tuple(y.shape),
                    t_get,
                    t_aug,
                    t_to,
                    t_fwd,
                    t_bwd,
                    t_batch,
                    _cuda_mem_stats(device),
                )
                debug_train_logged += 1
        train_losses[iepoch] /= nimg_per_epoch

        if iepoch == 5 or iepoch % 10 == 0:
            lavgt = 0.
            lavgct = 0.
            if test_data is not None or test_files is not None:
                np.random.seed(42)
                if nimg_test != nimg_test_per_epoch:
                    rperm = np.random.choice(np.arange(0, nimg_test),
                                             size=(nimg_test_per_epoch,), p=test_probs)
                else:
                    rperm = np.random.permutation(np.arange(0, nimg_test))
                debug_eval_logged = 0
                for ibatch in range(0, len(rperm), batch_size):
                    with torch.no_grad():
                        t0_eval_batch = time.perf_counter()
                        net.eval()
                        inds = rperm[ibatch:ibatch + batch_size]
                        t0_get = time.perf_counter()
                        imgs, lbls = _get_batch(inds, data=test_data,
                                                labels=test_labels, files=test_files,
                                                labels_files=test_labels_files,
                                                **kwargs)
                        t_get = time.perf_counter() - t0_get
                        diams = np.array([diam_test[i] for i in inds])
                        rsc = diams / diam_mean_value if rescale else np.ones(
                            len(diams), "float32")
                        t0_aug = time.perf_counter()
                        imgi, lbl = random_rotate_and_resize(
                            imgs, Y=lbls, rescale=rsc, scale_range=scale_range,
                            xy=(bsize, bsize))[:2]
                        t_aug = time.perf_counter() - t0_aug
                        
                        # print(f">> after augmentations label ids are {np.unique(lbl[0][1])}")
                        t0_to = time.perf_counter()
                        X = torch.from_numpy(imgi).to(device)
                        lbl = torch.from_numpy(lbl).to(device)
                        t_to = time.perf_counter() - t0_to
                        t0_fwd = time.perf_counter()
                        y = net(X)[0]
                        loss = _loss_fn_seg(lbl, y, device)
                        loss3 = None
                        if y.shape[1] > 3:
                            loss3 = _loss_fn_class(lbl, y, class_weights=class_weights)
                            loss = loss + loss3            
                        t_fwd = time.perf_counter() - t0_fwd
                        test_loss = loss.item()
                        test_class_loss = loss3.item() if loss3 is not None else 0.0
                        test_class_loss *= len(imgi)
                        test_loss *= len(imgi)
                        lavgct += test_class_loss
                        lavgt += test_loss
                        if debug_train and debug_eval_logged < debug_steps:
                            t_eval = time.perf_counter() - t0_eval_batch
                            train_logger.info(
                                "TRAIN_DEBUG: eval epoch=%s batch=%s size=%s xshape=%s yshape=%s "
                                "t_get=%.3fs t_aug=%.3fs t_to=%.3fs t_fwd=%.3fs t_total=%.3fs %s",
                                iepoch + 1,
                                (ibatch // batch_size) + 1,
                                len(inds),
                                tuple(X.shape),
                                tuple(y.shape),
                                t_get,
                                t_aug,
                                t_to,
                                t_fwd,
                                t_eval,
                                _cuda_mem_stats(device),
                            )
                            debug_eval_logged += 1
                lavgt /= len(rperm)
                lavgct /= len(rperm)
                test_losses[iepoch] = lavgt
                test_loss_epoch = lavgt
                test_class_loss_epoch = lavgct
                # simple early stopping on validation loss
                if early_stop:
                    if lavgt + min_delta < best_val:
                        best_val = lavgt
                        es_counter = 0
                    else:
                        es_counter += 1
                        if es_counter >= patience:
                            train_logger.info(f">>> early stopping at epoch {iepoch}")
                            # save current model before exiting early
                            try:
                                net.save_model(filename)
                            except Exception as e:
                                train_logger.warning(f"could not save model on early stop: {e}")
                            train_losses = train_losses[:iepoch + 1]
                            test_losses = test_losses[:iepoch + 1]
                            if progress_callback is not None:
                                try:
                                    progress_callback(100, "done", "Training complete (early stop)")
                                except Exception:
                                    pass
                            return filename, train_losses, test_losses
            lavg /= nsum
            lavgc /= nsum
            if debug_train:
                train_logger.info(
                    "TRAIN_DEBUG: epoch=%s summary train_loss=%.4f test_loss=%s %s",
                    iepoch + 1,
                    train_losses[iepoch],
                    f"{test_loss_epoch:.4f}" if test_loss_epoch is not None else "n/a",
                    _cuda_mem_stats(device),
                )
            print(
                f"{iepoch}, train_class_loss = {lavgc:.4f},train_loss={lavg:.4f}, test_class_loss = {lavgct:.4f},test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
            )
            lavg,lavgc, nsum = 0, 0,0

        if progress_callback is not None:
            try:
                pct = int(((iepoch + 1) / max(1, n_epochs)) * 100)
                msg = f"Epoch {iepoch + 1}/{n_epochs} train_loss={train_losses[iepoch]:.4f}"
                if test_loss_epoch is not None:
                    msg = f"{msg} test_loss={test_loss_epoch:.4f}"
                    if test_class_loss_epoch:
                        msg = f"{msg} test_class_loss={test_class_loss_epoch:.4f}"
                progress_callback(pct, "train", msg)
            except Exception:
                pass

        if iepoch == n_epochs - 1 or (iepoch % save_every == 0 and iepoch != 0):
            if save_each and iepoch != n_epochs - 1:  #separate files as model progresses
                filename0 = str(filename) + f"_epoch_{iepoch:04d}"
            else:
                filename0 = filename
            print(f"saving network parameters to {filename0}")
            net.save_model(filename0)
    
    net.save_model(filename)
    if progress_callback is not None:
        try:
            progress_callback(100, "done", "Training complete")
        except Exception:
            pass

    return filename, train_losses, test_losses
