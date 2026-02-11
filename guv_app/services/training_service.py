
import time
import os
import numpy as np
import logging
from pathlib import Path
import torch
from torch import nn
from tqdm import trange

from cellpose import io, utils, models, dynamics, train, remote_config, vit_sam
from cellpose.transforms import normalize_img, random_rotate_and_resize
from cellpose.semantic_class_weights import (
    compute_class_weights_from_class_maps,
    infer_semantic_nclasses_from_net,
)
from cellpose.training_mode_utils import configure_trainable_params

from guv_app.data_models.configs import TrainingConfig
from guv_app.data_models.results import TrainingResult

train_logger = logging.getLogger(__name__)

# All of the functions from cellpose/train.py are moved here to make the service self-contained.

def _loss_fn_class(lbl, y, class_weights=None):
    ncls = y.shape[1] - 3 if y.ndim >= 2 else 0
    if ncls <= 0:
        return 0. * y.sum()
    if class_weights is not None:
        try:
            if hasattr(class_weights, "shape") and class_weights.shape[0] != ncls:
                class_weights = None
        except Exception:
            class_weights = None
    if class_weights is not None and not torch.is_tensor(class_weights):
        try:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=y.device)
        except Exception:
            class_weights = None
    if lbl.ndim >= 3 and lbl.shape[1] > 1:
        tgt = torch.round(lbl[:, 1]).long()
    else:
        tgt = torch.round(lbl[:, 0]).long() if lbl.ndim >= 3 else torch.zeros_like(y[:, -1], dtype=torch.long)
    tgt = tgt.clamp(min=0, max=max(0, ncls - 1))
    criterion3 = nn.CrossEntropyLoss(reduction="mean", weight=class_weights)
    return criterion3(y[:, :-3], tgt)

def _loss_fn_seg(lbl, y, device):
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    if lbl.shape[1] >= 2:
        veci = 5. * lbl[:, -2:]
    else:
        veci = torch.zeros_like(y[:, -3:-1])
    loss = criterion(y[:, -3:-1], veci)
    loss /= 2.
    if lbl.shape[1] >= 3:
        cp_lbl = (lbl[:, -3] > 0.5).float()
    else:
        cp_lbl = (lbl[:, 0] > 0.5).float()
    loss2 = criterion2(y[:, -1], cp_lbl)
    loss = loss + loss2
    return loss

def _reshape_norm(data, channel_axis=None, normalize_params={"normalize": False}):
    if (np.array([td.ndim!=3 for td in data]).sum() > 0 or
        np.array([td.shape[0]!=3 for td in data]).sum() > 0):
        data_new = []
        for td in data:
            if td.ndim == 3:
                channel_axis0 = channel_axis if channel_axis is not None else np.array(td.shape).argmin()
                td = np.moveaxis(td, channel_axis0, 0)
                td = td[:3]
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


def _insert_class_map(flow_labels, class_maps):
    if not flow_labels:
        return flow_labels
    # If class maps are provided for semantic training, ensure every label carries
    # a class channel so augmentation sees a consistent channel count.
    if not class_maps:
        class_maps = [None] * len(flow_labels)
    updated = []
    for idx, lbl in enumerate(flow_labels):
        cmap = class_maps[idx] if idx < len(class_maps) else None
        try:
            if lbl.ndim != 3:
                updated.append(lbl)
                continue
            if cmap is not None:
                cmap = np.squeeze(cmap)
            if cmap is None or getattr(cmap, "ndim", 0) != 2 or tuple(cmap.shape) != tuple(lbl.shape[1:]):
                if lbl.shape[0] >= 5:
                    # Already has a class channel.
                    updated.append(lbl)
                    continue
                # Missing/invalid class map: add background-only channel to keep shape stable.
                bg_class = np.zeros(lbl.shape[1:], dtype=np.int64)
                combined = np.concatenate((lbl[:1], bg_class[np.newaxis, ...], lbl[1:]), axis=0)
                updated.append(combined)
            else:
                cmap = np.rint(cmap).astype(np.int64, copy=False)
                cmap = np.clip(cmap, 0, max(0, int(np.max(cmap))))
                if lbl.shape[0] >= 5:
                    combined = np.concatenate((lbl[:1], cmap[np.newaxis, ...], lbl[2:]), axis=0)
                    updated.append(combined)
                else:
                    combined = np.concatenate((lbl[:1], cmap[np.newaxis, ...], lbl[1:]), axis=0)
                    updated.append(combined)
        except Exception:
            updated.append(lbl)
    return updated


def _initialize_class_net(nclasses=2, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = remote_config.load_remote_config()
    cpsam_path = (
        cfg.get("cpsam_model_path")
        or os.environ.get("CELLPOSE_CPSAM_MODEL")
        or models.cache_CPSAM_model_path()
    )
    net = vit_sam.Transformer(rdrop=0.4).to(device)
    net.load_model(cpsam_path, device=device, strict=False)

    ps = 8
    nout = 3
    w0 = net.out.weight.data.detach().clone()
    b0 = net.out.bias.data.detach().clone()
    net.out = nn.Conv2d(256, (nout + nclasses) * ps**2, kernel_size=1).to(device)
    i = 0
    net.out.weight.data[i * ps**2 : (i + 1) * ps**2] = -0.5 * w0[(nout - 1) * ps**2 : nout * ps**2]
    net.out.bias.data[i * ps**2 : (i + 1) * ps**2] = b0[(nout - 1) * ps**2 : nout * ps**2]
    for i in range(1, nclasses):
        net.out.weight.data[i * ps**2 : (i + 1) * ps**2] = 0.5 * w0[(nout - 1) * ps**2 : nout * ps**2]
        net.out.bias.data[i * ps**2 : (i + 1) * ps**2] = b0[(nout - 1) * ps**2 : nout * ps**2]
    net.out.weight.data[-(nout * ps**2) :] = w0
    net.out.bias.data[-(nout * ps**2) :] = b0
    net.W2 = nn.Parameter(
        torch.eye((nout + nclasses) * ps**2).reshape((nout + nclasses) * ps**2, nout + nclasses, ps, ps),
        requires_grad=False,
    )
    net.to(device)
    return net


def _train_seg_with_progress(
    net,
    device,
    train_data,
    train_labels,
    diam_train,
    test_data,
    test_labels,
    diam_test,
    config,
    class_weights,
    normalize_params,
    progress_callback,
):
    nimg = len(train_data)
    LR = np.linspace(0, config.learning_rate, 10)
    LR = np.append(LR, config.learning_rate * np.ones(max(0, config.n_epochs - 10)))

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    train_losses = np.zeros(config.n_epochs)
    test_losses = np.zeros(config.n_epochs)
    for iepoch in range(config.n_epochs):
        np.random.seed(iepoch)
        rperm = np.random.permutation(np.arange(0, nimg))

        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]

        lavg, nsum = 0, 0
        lcls, lseg = 0.0, 0.0
        for k in range(0, nimg, config.batch_size):
            kend = min(k + config.batch_size, nimg)
            inds = rperm[k:kend]
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels, normalize_params=normalize_params)

            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / net.diam_mean.item()
            rescale = rsc if config.rescale else None

            imgi, lbl = random_rotate_and_resize(
                imgs,
                Y=lbls,
                rescale=rescale,
                scale_range=config.scale_range,
                xy=(config.bsize, config.bsize),
            )[:2]

            X = torch.from_numpy(imgi).to(device)
            lbl = torch.from_numpy(lbl).to(device)
            y = net(X)[0]
            loss = _loss_fn_seg(lbl, y, device)
            loss3 = _loss_fn_class(lbl, y, class_weights=class_weights)
            loss = loss + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item() * len(imgi)
            lavg += train_loss
            nsum += len(imgi)
            lseg += loss.item() * len(imgi) if hasattr(loss, "item") else 0.0
            lcls += loss3.item() * len(imgi) if hasattr(loss3, "item") else 0.0

        train_losses[iepoch] = lavg / max(1, nsum)

        test_loss_value = None
        if test_data is not None and test_labels is not None and len(test_data) > 0:
            tavg, tnsum = 0, 0
            tcls, tseg = 0.0, 0.0
            for k in range(0, len(test_data), config.batch_size):
                kend = min(k + config.batch_size, len(test_data))
                inds = np.arange(k, kend)
                imgs, lbls = _get_batch(inds, data=test_data, labels=test_labels, normalize_params=normalize_params)
                tdiams = np.array([diam_test[i] for i in inds]) if diam_test is not None else np.ones(len(inds))
                rsc = tdiams / net.diam_mean.item()
                rescale = rsc if config.rescale else None
                imgi, lbl = random_rotate_and_resize(
                    imgs,
                    Y=lbls,
                    rescale=rescale,
                    scale_range=config.scale_range,
                    xy=(config.bsize, config.bsize),
                )[:2]
                X = torch.from_numpy(imgi).to(device)
                lbl = torch.from_numpy(lbl).to(device)
                y = net(X)[0]
                loss = _loss_fn_seg(lbl, y, device)
                loss3 = _loss_fn_class(lbl, y, class_weights=class_weights)
                loss = loss + loss3
                tavg += loss.item() * len(imgi)
                tnsum += len(imgi)
                tseg += loss.item() * len(imgi) if hasattr(loss, "item") else 0.0
                tcls += loss3.item() * len(imgi) if hasattr(loss3, "item") else 0.0
            test_losses[iepoch] = tavg / max(1, tnsum)
            test_loss_value = test_losses[iepoch]

        if progress_callback:
            progress_callback(iepoch, config.n_epochs, train_losses[iepoch], test_loss_value or 0.0)

        if iepoch % 5 == 0:
            train_logger.info(
                "GUI_INFO: epoch %s/%s train_loss=%.4f train_class=%.4f",
                iepoch + 1,
                config.n_epochs,
                train_losses[iepoch],
                lcls / max(1, nsum),
            )
            if test_data is not None and test_labels is not None and len(test_data) > 0:
                train_logger.info(
                    "GUI_INFO: epoch %s/%s test_loss=%.4f test_class=%.4f",
                    iepoch + 1,
                    config.n_epochs,
                    test_losses[iepoch],
                    tcls / max(1, tnsum),
                )

    return train_losses, test_losses

def _get_batch(inds, data=None, labels=None, files=None, labels_files=None,
               normalize_params={"normalize": False}):
    if data is None:
        lbls = None
        imgs = [io.imread(files[i]) for i in inds]
        imgs = _reshape_norm(imgs, normalize_params=normalize_params)
        if labels_files is not None:
            lbls = [io.imread(labels_files[i])[1:] for i in inds]
    else:
        imgs = [data[i] for i in inds]
        lbls = [labels[i] for i in inds]
        if normalize_params.get("normalize"):
            imgs = _reshape_norm(imgs, normalize_params=normalize_params)
    return imgs, lbls

def _process_train_test(train_data=None, train_labels=None, train_files=None,
                        train_labels_files=None, train_probs=None, test_data=None,
                        test_labels=None, test_files=None, test_labels_files=None,
                        test_probs=None, load_files=True, min_train_masks=5,
                        compute_flows=False, normalize_params={"normalize": False}, 
                        channel_axis=None, device=None, class_maps=None, test_class_maps=None,
                        masks_for_diam=None, test_masks_for_diam=None):
    if device == None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else None
    
    if train_data is not None and train_labels is not None:
        nimg = len(train_data)
        nimg_test = len(test_data) if test_data is not None else None
    else:
        nimg = len(train_files)
        if train_labels_files is None:
            train_labels_files = [os.path.splitext(str(tf))[0] + "_flows.tif" for tf in train_files]
            train_labels_files = [tf for tf in train_labels_files if os.path.exists(tf)]
        if (test_data is not None or test_files is not None) and test_labels_files is None:
            test_labels_files = [os.path.splitext(str(tf))[0] + "_flows.tif" for tf in test_files]
            test_labels_files = [tf for tf in test_labels_files if os.path.exists(tf)]
        if not load_files:
            train_logger.info(">>> using files instead of loading dataset")
        else:
            train_logger.info(">>> loading images and labels")
            train_data = [io.imread(train_files[i]) for i in trange(nimg)]
            train_labels = [io.imread(train_labels_files[i]) for i in trange(nimg)]
        nimg_test = len(test_files) if test_files is not None else None
        if load_files and nimg_test:
            test_data = [io.imread(test_files[i]) for i in trange(nimg_test)]
            test_labels = [io.imread(test_labels_files[i]) for i in trange(nimg_test)]

    if ((train_labels is not None and nimg != len(train_labels)) or
        (train_labels_files is not None and nimg != len(train_labels_files))):
        raise ValueError("train data and labels not same length")
    
    if masks_for_diam is None:
        masks_for_diam = train_labels
    if train_labels is not None:
        train_labels = dynamics.labels_to_flows(train_labels, files=train_files, device=device)
        if class_maps:
            train_labels = _insert_class_map(train_labels, class_maps)
        if test_labels is not None:
            test_labels = dynamics.labels_to_flows(test_labels, files=test_files, device=device)
            if test_class_maps:
                test_labels = _insert_class_map(test_labels, test_class_maps)

    nmasks = np.zeros(nimg)
    diam_train = np.zeros(nimg)
    train_logger.info(">>> computing diameters")
    for k in trange(nimg):
        if masks_for_diam is not None:
            tl = masks_for_diam[k]
            if getattr(tl, "ndim", 0) >= 3:
                tl = tl[0]
        else:
            tl = io.imread(train_labels_files[k])[0]
        diam_train[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam_train[diam_train < 5] = 5.
    
    if min_train_masks > 0:
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            if train_data is not None:
                train_data = [train_data[i] for i in ikeep]
                train_labels = [train_labels[i] for i in ikeep]
            diam_train = diam_train[ikeep]
            nimg = len(train_data)

    return (train_data, train_labels, None, None, None, diam_train, None, None, None, None, None, None, False)


class TrainingService:
    def __init__(self, net=None):
        self.net = net

    def start_training(
        self,
        config: TrainingConfig,
        progress_callback=None,
        train_data=None,
        train_labels=None,
        test_data=None,
        test_labels=None,
        class_maps=None,
        test_class_maps=None,
        flow_labels=None,
        test_flow_labels=None,
    ) -> TrainingResult:
        
        # Most of this is from cellpose.train.train_seg
        
        self.net.train()
        device = getattr(self.net, "device", None)
        if device is None:
            try:
                device = next(self.net.parameters()).device
            except Exception:
                device = torch.device("cpu")

        normalize_params = models.normalize_default
        normalize_params["normalize"] = True

        try:
            mode_info = configure_trainable_params(
                self.net,
                use_lora=bool(config.use_lora),
                lora_blocks=(
                    config.lora_blocks
                    if config.lora_blocks is not None
                    else config.unfreeze_blocks
                ),
                unfreeze_blocks=config.unfreeze_blocks,
                logger=train_logger,
            )
            if config.use_lora and mode_info.get("lora_info") is not None:
                lora_info = mode_info["lora_info"]
                train_logger.info(
                    "GUI_INFO: LoRA injected into last %s/%s encoder blocks, converted_linear_layers=%s",
                    lora_info.get("applied_blocks"),
                    lora_info.get("total_blocks"),
                    lora_info.get("converted_linear_layers"),
                )
                try:
                    rep = models.lora_trainability_report(self.net)
                    train_logger.info(
                        "GUI_INFO: LoRA params total=%s trainable=%s lora_trainable=%s out_trainable=%s",
                        rep["total_params"],
                        rep["trainable_params"],
                        rep["lora_trainable_params"],
                        rep["out_trainable_params"],
                    )
                    if rep["unexpected_encoder_trainable"]:
                        train_logger.warning(
                            "GUI_WARN: unexpected trainable encoder params detected in LoRA mode (n=%s): %s",
                            len(rep["unexpected_encoder_trainable"]),
                            rep["unexpected_encoder_trainable"][:10],
                        )
                except Exception:
                    pass
        except Exception:
            pass

        min_train_masks = 0
        if config.min_train_masks not in (0, None):
            train_logger.info(
                "Remote alignment: ignoring min_train_masks=%s and using 0.",
                config.min_train_masks,
            )
        train_labels_for_training = train_labels
        test_labels_for_training = test_labels

        if class_maps:
            if train_labels_for_training is not None and (
                not flow_labels or any(lbl is None for lbl in flow_labels)
            ):
                train_labels_for_training = dynamics.labels_to_flows(
                    train_labels_for_training, files=config.train_files, device=device
                )
            train_labels_for_training = _insert_class_map(
                train_labels_for_training, class_maps
            )
        if test_class_maps:
            if test_labels_for_training is not None and (
                not test_flow_labels or any(lbl is None for lbl in test_flow_labels)
            ):
                test_labels_for_training = dynamics.labels_to_flows(
                    test_labels_for_training, files=config.test_files, device=device
                )
            test_labels_for_training = _insert_class_map(
                test_labels_for_training, test_class_maps
            )

        if not hasattr(self.net, "diam_mean") or self.net.diam_mean is None or not torch.is_tensor(self.net.diam_mean):
            self.net.diam_mean = torch.nn.Parameter(torch.tensor([30.0]), requires_grad=False)
        if not hasattr(self.net, "diam_labels") or self.net.diam_labels is None or not torch.is_tensor(self.net.diam_labels):
            self.net.diam_labels = torch.nn.Parameter(torch.tensor([30.0]), requires_grad=False)

        try:
            from cellpose import models as cp_models
            default_dir = cp_models.MODEL_DIR
        except Exception:
            default_dir = Path.cwd() / "models"
        save_dir = Path(config.save_path) if config.save_path else default_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        nclasses_inferred = infer_semantic_nclasses_from_net(self.net)
        valid_class_maps = []
        if class_maps:
            for cm in class_maps:
                if cm is None:
                    continue
                try:
                    cm = np.squeeze(cm)
                    if cm.ndim == 2:
                        valid_class_maps.append(np.rint(cm).astype(np.int64, copy=False))
                except Exception:
                    continue
        max_class_id = max((int(np.max(cm)) for cm in valid_class_maps), default=-1)
        class_weights = compute_class_weights_from_class_maps(
            valid_class_maps, nclasses=nclasses_inferred
        )
        train_logger.info(
            "GUI_INFO: class-weight inputs: valid_class_maps=%s max_class_id=%s inferred_nclasses=%s",
            len(valid_class_maps),
            max_class_id,
            nclasses_inferred,
        )
        if class_weights is not None:
            train_logger.info(
                "GUI_INFO: class weights (%s classes incl. bg): %s",
                int(len(class_weights)),
                class_weights,
            )
            train_logger.info(
                "GUI_INFO: class-weight outputs: weight_vector_length=%s",
                int(len(class_weights)),
            )
        else:
            train_logger.info(
                "GUI_INFO: class-weight outputs: weight_vector_length=0 (unweighted CE fallback)"
            )
        if class_maps:
            try:
                max_classes = [int(np.max(cm)) for cm in class_maps if cm is not None]
                if max_classes:
                    train_logger.info(
                        "GUI_INFO: class_map max values (sample): %s",
                        max_classes[:10],
                    )
            except Exception:
                pass
        try:
            if hasattr(self.net, "out"):
                out_weight = self.net.out.weight
                train_logger.info(
                    "GUI_INFO: net output weight shape: %s",
                    tuple(out_weight.shape),
                )
        except Exception:
            pass
        train_logger.info("GUI_INFO: name of new model: %s", config.model_name)

        test_files = config.test_files if config.test_files else None
        if not test_files:
            test_data = None
            test_labels = None
            test_labels_for_training = None
        model_path, train_losses, test_losses = train.train_seg(
            self.net,
            train_data=train_data,
            train_labels=train_labels_for_training,
            test_data=test_data,
            test_labels=test_labels_for_training,
            test_files=test_files,
            normalize=normalize_params,
            min_train_masks=min_train_masks,
            batch_size=config.batch_size,
            bsize=config.bsize,
            rescale=config.rescale,
            scale_range=config.scale_range,
            save_path=str(save_dir),
            nimg_per_epoch=len(train_data),
            nimg_test_per_epoch=len(test_data) if test_data else 0,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            n_epochs=config.n_epochs,
            early_stop=False,
            model_name=config.model_name,
            class_weights=class_weights,
            progress_callback=progress_callback,
        )

        if config.use_lora:
            try:
                self.net.load_model(model_path, device=device)
                models.merge_and_remove_lora(self.net)
                torch.save(self.net.state_dict(), model_path)
                train_logger.info("GUI_INFO: merged LoRA adapters into saved model: %s", model_path)
            except Exception as exc:
                train_logger.warning("GUI_WARN: failed to merge LoRA adapters into saved model (%s)", exc)

        return TrainingResult(
            model_path=str(model_path),
            train_losses=train_losses.tolist() if hasattr(train_losses, "tolist") else train_losses,
            test_losses=test_losses.tolist() if hasattr(test_losses, "tolist") else test_losses,
        )
