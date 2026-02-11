# Cellpose / cpgrpc code context

This file lists **every class, function, and class method** found under `cellpose/` and `cpgrpc/`, with a one‑sentence description.

## `cellpose/__init__.py`

_No top-level classes or functions found._

## `cellpose/__main__.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

- `main` (function): Run cellpose from command line
- `_train_cellposemodel_cli` (function): Implements  train cellposemodel cli.
- `_evaluate_cellposemodel_cli` (function): Implements  evaluate cellposemodel cli.

## `cellpose/cli.py`

**Module:** Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu and Michael Rariden.

- `get_arg_parser` (function): Parses command line arguments for cellpose main function  Note: this function has to be in a separate file to allow autodoc to work for CLI.

## `cellpose/contrib/distributed_segmentation.py`

- `numpy_array_to_zarr` (function): Store an in memory numpy array to disk as a chunked Zarr array  Parameters ---------- write_path : string     Filepath where Zarr array will be created  array : numpy.ndarray     The already loaded in-memory numpy array to store as zarr  chunks : tuple, must be array.ndim length     How the array will be chunked in the Zarr array  Returns ------- zarr.core.Array     A read+write reference to the zarr array on disk
- `wrap_folder_of_tiffs` (function): Wrap a folder of tiff files with a zarr array without duplicating data.
- `_config_path` (function): Add config directory path to config filename
- `_modify_dask_config` (function): Modifies dask config dictionary, but also dumps modified config to disk as a yaml file in ~/.config/dask/.
- `_remove_config_file` (function): Removes a config file from disk
- `myLocalCluster` (class): This is a thin wrapper extending dask.distributed.LocalCluster to set configs before the cluster or workers are initialized.
- `myLocalCluster.__init__` (method): Initializes the object and sets up its internal state.
- `myLocalCluster.__enter__` (method): Implements Python dunder behavior for __enter__.
- `myLocalCluster.__exit__` (method): Implements Python dunder behavior for __exit__.
- `janeliaLSFCluster` (class): This is a thin wrapper extending dask_jobqueue.LSFCluster, which in turn extends dask.distributed.SpecCluster.
- `janeliaLSFCluster.__init__` (method): Initializes the object and sets up its internal state.
- `janeliaLSFCluster.__enter__` (method): Implements Python dunder behavior for __enter__.
- `janeliaLSFCluster.__exit__` (method): Implements Python dunder behavior for __exit__.
- `janeliaLSFCluster.adapt_cluster` (method): Implements adapt cluster.
- `janeliaLSFCluster.change_worker_attributes` (method): WARNING: this function is dangerous if you don't know what you're doing.
- `cluster` (function): This decorator ensures a function will run inside a cluster as a context manager.
- `process_block` (function): Preprocess and segment one block, of many, with eventual merger of all blocks in mind.
- `read_preprocess_and_segment` (function): Read block from zarr array, run all preprocessing steps, run cellpose
- `remove_overlaps` (function): overlaps only there to provide context for boundary voxels and can be removed after segmentation is complete reslice array to remove the overlaps
- `bounding_boxes_in_global_coordinates` (function): bounding boxes (tuples of slices) are super useful later best to compute them now while things are distributed
- `get_nblocks` (function): Given a shape and blocksize determine the number of blocks per axis
- `global_segment_ids` (function): pack the block index into the segment IDs so they are globally unique.
- `block_faces` (function): slice faces along every axis
- `distributed_eval` (function): Evaluate a cellpose model on overlapping blocks of a big image.
- `get_block_crops` (function): Given a voxel grid shape, blocksize, and overlap size, construct tuples of slices for every block; optionally only include blocks that contain foreground in the mask.
- `determine_merge_relabeling` (function): Determine boundary segment mergers, remap all label IDs to merge and put all label IDs in range [1..N] for N global segments found
- `adjacent_faces` (function): Find faces which touch and pair them together in new data structure
- `block_face_adjacency_graph` (function): Shrink labels in face plane, then find which labels touch across the face boundary
- `shrink_labels` (function): Shrink labels in plane by some distance from their boundary
- `merge_all_boxes` (function): Merge all boxes that map to the same box_ids
- `merge_boxes` (function): Take union of two or more parallelpipeds

## `cellpose/core.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

- `use_gpu` (function): Check if GPU is available for use.
- `_use_gpu_torch` (function): Checks if CUDA or MPS is available and working with PyTorch.
- `assign_device` (function): Assigns the device (CPU or GPU or mps) to be used for computation.
- `_to_device` (function): Converts the input tensor or numpy array to the specified device.
- `_from_device` (function): Converts a PyTorch tensor from the device to a NumPy array on the CPU.
- `_forward` (function): Converts images to torch tensors, runs the network model, and returns numpy arrays.
- `run_net` (function): Run network on stack of images.
- `run_3D` (function): Run network on image z-stack.

## `cellpose/denoise.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

- `deterministic` (function): set random seeds to create test data
- `loss_fn_rec` (function): loss function between true labels lbl and prediction y
- `loss_fn_seg` (function): loss function between true labels lbl and prediction y
- `get_sigma` (function): Calculates the correlation matrices across channels for the perceptual loss.
- `imstats` (function): Calculates the image correlation matrices for the perceptual loss.
- `loss_fn_per` (function): Calculates the perceptual loss function for image restoration.
- `test_loss` (function): Calculates the test loss for image restoration tasks.
- `train_loss` (function): Calculates the train loss for image restoration tasks.
- `img_norm` (function): Normalizes the input image by subtracting the 1st percentile and dividing by the difference between the 99th and 1st percentiles.
- `add_noise` (function): Adds noise to the input image.
- `random_rotate_and_resize_noise` (function): Applies random rotation, resizing, and noise to the input data.
- `one_chan_cellpose` (function): Creates a Cellpose network with a single input channel.
- `CellposeDenoiseModel` (class): model to run Cellpose and Image restoration
- `CellposeDenoiseModel.__init__` (method): Initializes the object and sets up its internal state.
- `CellposeDenoiseModel.eval` (method): Restore array or list of images using the image restoration model, and then segment.
- `DenoiseModel` (class): DenoiseModel class for denoising images using Cellpose denoising model.
- `DenoiseModel.__init__` (method): Initializes the object and sets up its internal state.
- `DenoiseModel.eval` (method): Restore array or list of images using the image restoration model.
- `DenoiseModel._eval` (method): Run image restoration model on a single channel.
- `train` (function): Trains a model using the provided data and hyperparameters.
- `seg_train_noisy` (function): train function uses loss function model.loss_fn in models.py  (data should already be normalized)

## `cellpose/dynamics.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

- `_extend_centers_gpu` (function): Runs diffusion on GPU to generate flows for training images or quality control.
- `center_of_mass` (function): Implements center of mass.
- `get_centers` (function): Retrieves centers.
- `masks_to_flows_gpu` (function): Convert masks to flows using diffusion from center pixel.
- `masks_to_flows_gpu_3d` (function): Convert masks to flows using diffusion from center pixel.
- `labels_to_flows` (function): Converts labels (list of masks or flows) to flows for training model.
- `flow_error` (function): Error in flows from predicted masks vs flows predicted by network run on image.
- `steps_interp` (function): Run dynamics of pixels to recover masks in 2D/3D, with interpolation between pixel values.
- `follow_flows` (function): Run dynamics to recover masks in 2D or 3D.
- `remove_bad_flow_masks` (function): Remove masks which have inconsistent flows.
- `max_pool1d` (function): memory efficient max_pool thanks to Mark Kittisopikul   for stride=1, padding=kernel_size//2, requires odd kernel_size >= 3
- `max_pool_nd` (function): memory efficient max_pool in 2d or 3d
- `get_masks_torch` (function): Create masks using pixel convergence after running dynamics.
- `resize_and_compute_masks` (function): Compute masks using dynamics from dP and cellprob, and resizes masks if resize is not None.
- `compute_masks` (function): Compute masks using dynamics from dP and cellprob.

## `cellpose/export.py`

**Module:** Auxiliary module for bioimageio format export  Example usage:  ```bash #!/bin/bash  # Define default paths and parameters DEFAULT_CHANNELS="1 0" DEFAULT_PATH_PRETRAINED_MODEL="/home/qinyu/models/cp/cellpose_residual_on_style_on_concatenation_off_1135_rest_2023_05_04_23_41_31.252995" DEFAULT_PATH_README="/home/qinyu/models/cp/README.md" DEFAULT_LIST_PATH_COVER_IMAGES="/home/qinyu/images/cp/cellpose_raw_and_segmentation.jpg /home/qinyu/images/cp/cellpose_raw_and_probability.jpg /home/qinyu/images/cp/cellpose_raw.jpg" DEFAULT_MODEL_ID="philosophical-panda" DEFAULT_MODEL_ICON="🐼" DEFAULT_MODEL_VERSION="0.1.0" DEFAULT_MODEL_NAME="My Cool Cellpose" DEFAULT_MODEL_DOCUMENTATION="A cool Cellpose model trained for my cool dataset." DEFAULT_MODEL_AUTHORS='[{"name": "Qin Yu", "affiliation": "EMBL", "github_user": "qin-yu", "orcid": "0000-0002-4652-0795"}]' DEFAULT_MODEL_CITE='[{"text": "For more details of the model itself, see the manuscript", "doi": "10.1242/dev.202800", "url": null}]' DEFAULT_MODEL_TAGS="cellpose 3d 2d" DEFAULT_MODEL_LICENSE="MIT" DEFAULT_MODEL_REPO="https://github.com/kreshuklab/go-nuclear"  # Run the Python script with default parameters python export.py     --channels $DEFAULT_CHANNELS     --path_pretrained_model "$DEFAULT_PATH_PRETRAINED_MODEL"     --path_readme "$DEFAULT_PATH_README"     --list_path_cover_images $DEFAULT_LIST_PATH_COVER_IMAGES     --model_version "$DEFAULT_MODEL_VERSION"     --model_name "$DEFAULT_MODEL_NAME"     --model_documentation "$DEFAULT_MODEL_DOCUMENTATION"     --model_authors "$DEFAULT_MODEL_AUTHORS"     --model_cite "$DEFAULT_MODEL_CITE"     --model_tags $DEFAULT_MODEL_TAGS     --model_license "$DEFAULT_MODEL_LICENSE"     --model_repo "$DEFAULT_MODEL_REPO" ```

- `download_and_normalize_image` (function): Download and normalize image.
- `load_bioimageio_cpnet_model` (function): Loads bioimageio cpnet model from disk or configuration.
- `descr_gen_input` (function): Implements descr gen input.
- `descr_gen_output_flow` (function): Implements descr gen output flow.
- `descr_gen_output_downsampled` (function): Implements descr gen output downsampled.
- `descr_gen_output_style` (function): Implements descr gen output style.
- `descr_gen_arch` (function): Implements descr gen arch.
- `descr_gen_documentation` (function): Implements descr gen documentation.
- `package_to_bioimageio` (function): Package model description to BioImage.IO format.
- `parse_args` (function): Implements parse args.
- `main` (function): Entry point for running this module as a script.

## `cellpose/gui/gui.py`

_No top-level classes or functions found._

## `cellpose/gui/gui3d.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.

- `avg3d` (function): smooth value of c across nearby points (c is center of grid directly below point) b -- a -- b a -- c -- a b -- a -- b
- `interpZ` (function): find nearby planes and average their values using grid of points zfill is in ascending order
- `run` (function): Runs inference to produce predictions from inputs.
- `MainW_3d` (class): Defines the MainW_3d class and its behavior.
- `MainW_3d.__init__` (method): Initializes the object and sets up its internal state.
- `MainW_3d.add_mask` (method): Implements add mask.
- `MainW_3d.move_in_Z` (method): Implements move in z.
- `MainW_3d.make_orthoviews` (method): Implements make orthoviews.
- `MainW_3d.add_orthoviews` (method): Implements add orthoviews.
- `MainW_3d.remove_orthoviews` (method): Implements remove orthoviews.
- `MainW_3d.update_crosshairs` (method): Updates internal state based on new inputs.
- `MainW_3d.update_ortho` (method): Updates internal state based on new inputs.
- `MainW_3d.toggle_ortho` (method): Implements toggle ortho.
- `MainW_3d.plot_clicked` (method): Visualizes results for inspection.
- `MainW_3d.update_plot` (method): Updates internal state based on new inputs.
- `MainW_3d.keyPressEvent` (method): Implements key press event.
- `MainW_3d.update_ztext` (method): Updates internal state based on new inputs.

## `cellpose/gui/guiparts.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

- `stylesheet` (function): Implements stylesheet.
- `DarkPalette` (class): Class that inherits from pyqtgraph.QtGui.QPalette and renders dark colours for the application.
- `DarkPalette.__init__` (method): Initializes the object and sets up its internal state.
- `DarkPalette.setup` (method): Implements setup.
- `ModelButton` (class): Defines the ModelButton class and its behavior.
- `ModelButton.__init__` (method): Initializes the object and sets up its internal state.
- `ModelButton.press` (method): Implements press.
- `FilterButton` (class): Defines the FilterButton class and its behavior.
- `FilterButton.__init__` (method): Initializes the object and sets up its internal state.
- `FilterButton.press` (method): Implements press.
- `ObservableVariable` (class): Defines the ObservableVariable class and its behavior.
- `ObservableVariable.__init__` (method): Initializes the object and sets up its internal state.
- `ObservableVariable.set` (method): Use this method to get emit the value changing and update the ROI count
- `ObservableVariable.get` (method): Implements get.
- `ObservableVariable.__call__` (method): Makes the instance callable and forwards to the main execution logic.
- `ObservableVariable.reset` (method): Implements reset.
- `ObservableVariable.__iadd__` (method): Implements Python dunder behavior for __iadd__.
- `ObservableVariable.__radd__` (method): Implements Python dunder behavior for __radd__.
- `ObservableVariable.__add__` (method): Implements Python dunder behavior for __add__.
- `ObservableVariable.__isub__` (method): Implements Python dunder behavior for __isub__.
- `ObservableVariable.__str__` (method): Implements Python dunder behavior for __str__.
- `ObservableVariable.__lt__` (method): Implements Python dunder behavior for __lt__.
- `ObservableVariable.__gt__` (method): Implements Python dunder behavior for __gt__.
- `ObservableVariable.__eq__` (method): Implements Python dunder behavior for __eq__.
- `NormalizationSettings` (class): Defines the NormalizationSettings class and its behavior.
- `SegmentationSettings` (class): Container for gui settings.
- `SegmentationSettings.__init__` (method): Initializes the object and sets up its internal state.
- `SegmentationSettings.validate_normalization_range` (method): Implements validate normalization range.
- `SegmentationSettings.low_percentile` (method): Also validate the low input by returning 1.0 if text doesn't work
- `SegmentationSettings.high_percentile` (method): Also validate the high input by returning 99.0 if text doesn't work
- `SegmentationSettings.diameter` (method): Get the diameter from the diameter box, if box isn't a number return None
- `SegmentationSettings.flow_threshold` (method): Implements flow threshold.
- `SegmentationSettings.cellprob_threshold` (method): Implements cellprob threshold.
- `SegmentationSettings.max_size_fraction` (method): Implements max size fraction.
- `SegmentationSettings.niter` (method): Implements niter.
- `TrainWindow` (class): Defines the TrainWindow class and its behavior.
- `TrainWindow.__init__` (method): Initializes the object and sets up its internal state.
- `TrainWindow.accept` (method): Implements accept.
- `ExampleGUI` (class): Defines the ExampleGUI class and its behavior.
- `ExampleGUI.__init__` (method): Initializes the object and sets up its internal state.
- `HelpWindow` (class): Defines the HelpWindow class and its behavior.
- `HelpWindow.__init__` (method): Initializes the object and sets up its internal state.
- `TrainHelpWindow` (class): Defines the TrainHelpWindow class and its behavior.
- `TrainHelpWindow.__init__` (method): Initializes the object and sets up its internal state.
- `ViewBoxNoRightDrag` (class): Defines the ViewBoxNoRightDrag class and its behavior.
- `ViewBoxNoRightDrag.__init__` (method): Initializes the object and sets up its internal state.
- `ViewBoxNoRightDrag.keyPressEvent` (method): This routine should capture key presses in the current view box.
- `ImageDraw` (class): **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>` GraphicsObject displaying an image.
- `ImageDraw.__init__` (method): Initializes the object and sets up its internal state.
- `ImageDraw.mouseClickEvent` (method): Implements mouse click event.
- `ImageDraw.mouseDragEvent` (method): Implements mouse drag event.
- `ImageDraw.hoverEvent` (method): Implements hover event.
- `ImageDraw.create_start` (method): Implements create start.
- `ImageDraw.is_at_start` (method): Implements is at start.
- `ImageDraw.end_stroke` (method): Implements end stroke.
- `ImageDraw.tabletEvent` (method): Implements tablet event.
- `ImageDraw.drawAt` (method): Implements draw at.
- `ImageDraw.setDrawKernel` (method): Implements set draw kernel.

## `cellpose/gui/io.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

- `_training_dir_for_image` (function): Implements  training dir for image.
- `_tile_index_path` (function): Implements  tile index path.
- `_load_tile_index` (function): Implements  load tile index.
- `_write_tile_index` (function): Implements  write tile index.
- `_build_tile_index` (function): Implements  build tile index.
- `_ensure_tile_info` (function): Implements  ensure tile info.
- `_tile_seg_path` (function): Implements  tile seg path.
- `_assemble_tiled_seg` (function): Implements  assemble tiled seg.
- `_merge_tile_connected` (function): Implements  merge tile connected.
- `_save_tiled_sets` (function): Implements  save tiled sets.
- `_save_tiled_pred_files` (function): Implements  save tiled pred files.
- `_collect_tile_images` (function): Implements  collect tile images.
- `_ensure_optional_dependency` (function): Implements  ensure optional dependency.
- `_init_model_list` (function): Implements  init model list.
- `_add_model` (function): Implements  add model.
- `_remove_model` (function): Implements  remove model.
- `_get_train_set` (function): get training data and labels for images in current folder image_names
- `_load_image` (function): load image with filename; if None, open QFileDialog if image is grey change view to default to grey scale
- `_initialize_images` (function): format image for GUI  assumes image is Z x W x H x C
- `_load_seg` (function): load *_seg.npy with filename; if None, open QFileDialog
- `_load_masks` (function): load zeros-based masks (0=no cell, 1=cell 1, ...)
- `_masks_to_gui` (function): masks loaded into GUI
- `_save_png` (function): save masks to png or tiff (if 3D)
- `_save_flows` (function): save flows and cellprob to tiff
- `_save_rois` (function): save masks as rois in .zip file for ImageJ
- `_save_outlines` (function): Implements  save outlines.
- `_save_sets_with_check` (function): Save masks and update *_seg.npy file.
- `_save_sets` (function): save masks to *_seg.npy.
- `_save_sets_to_path` (function): save masks to a custom npy path without touching *_seg.npy
- `_convert_to_tifs` (function): Implements  convert to tifs.
- `_needs_stack_selection` (function): Implements  needs stack selection.
- `_select_stack_slice` (function): Implements  select stack slice.
- `_read_gui_image` (function): Implements  read gui image.

## `cellpose/gui/make_train.py`

- `main` (function): Entry point for running this module as a script.

## `cellpose/gui/menus.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

- `mainmenu` (function): Entry point for running this module as a script.
- `editmenu` (function): Implements editmenu.
- `modelmenu` (function): Implements modelmenu.
- `helpmenu` (function): Implements helpmenu.
- `add_masks_menu` (function): Implements add masks menu.

## `cellpose/io.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

- `logger_setup` (function): Implements logger setup.
- `check_dir` (function): Implements check dir.
- `outlines_to_text` (function): Implements outlines to text.
- `_load_seg` (function): Implements  load seg.
- `load_dax` (function): Loads dax from disk or configuration.
- `imread` (function): Read in an image file with tif or image file type supported by cv2.
- `_infer_channel_axis` (function): Heuristic to find channel axis.
- `_maybe_tile_image` (function): Split large images into non-overlapping tiles; optionally save each tile as a TIFF.
- `_prompt_overwrite_tiles` (function): Ask user if existing tiles for this source should be overwritten.
- `_maybe_save_tif_copy` (function): Save a TIFF copy to enable downstream labeling/training workflows.
- `_maybe_warn_non_tiff` (function): Show a GUI info box when opening non-TIFF containers (if GUI available).
- `_prompt_channel_split` (function): Prompt user to select channels to split; returns list of channel indices or None.
- `imread_2D` (function): Read in a 2D image file and convert it to a 3-channel image.
- `imread_3D` (function): Read in a 3D image file and convert it to have a channel axis last automatically.
- `remove_model` (function): remove model from .cellpose custom model list
- `add_model` (function): add model to .cellpose models folder to use with GUI or CLI
- `imsave` (function): Saves an image array to a file.
- `get_image_files` (function): Finds all images in a folder and its subfolders (if specified) with the given file extensions.
- `get_label_files` (function): Get the label files corresponding to the given image names and mask filter.
- `load_images_labels` (function): Loads images and corresponding labels from a directory.
- `load_train_test_data` (function): Loads training and testing data for a Cellpose model.
- `masks_flows_to_seg` (function): Save output of model eval to be loaded in GUI.
- `save_to_png` (function): deprecated (runs io.save_masks with png=True)  does not work for 3D images
- `save_rois` (function): save masks to .roi files in .zip archive for ImageJ/Fiji  Args:     masks (np.ndarray): masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels     file_name (str): name to save the .zip file to  Returns:     None
- `save_masks` (function): Save masks + nicely plotted segmentation image to png and/or tiff.

## `cellpose/metrics.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

- `mask_ious` (function): Return best-matched masks.
- `boundary_scores` (function): Calculate boundary precision, recall, and F-score.
- `aggregated_jaccard_index` (function): AJI = intersection of all matched masks / union of all masks   Args:     masks_true (list of np.ndarrays (int) or np.ndarray (int)):          where 0=NO masks; 1,2...
- `average_precision` (function): Average precision estimation: AP = TP / (TP + FP + FN)  This function is based heavily on the *fast* stardist matching functions (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)  Args:     masks_true (list of np.ndarrays (int) or np.ndarray (int)):          where 0=NO masks; 1,2...
- `_intersection_over_union` (function): Calculate the intersection over union of all mask pairs.
- `_true_positive` (function): Calculate the true positive at threshold th.

## `cellpose/models.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.

- `model_path` (function): Implements model path.
- `cache_CPSAM_model_path` (function): Implements cache cpsam model path.
- `get_user_models` (function): Retrieves user models.
- `CellposeModel` (class): Class representing a Cellpose model.
- `CellposeModel.__init__` (method): Initialize the CellposeModel.
- `CellposeModel.eval` (method): segment list of images x, or 4D array - Z x 3 x Y x X  Args:     x (list, np.ndarry): can be list of 2D/3D/4D images, or array of 2D/3D/4D images.
- `CellposeModel._resize_cellprob` (method): Resize cellprob array to specified dimensions for either 2D or 3D.
- `CellposeModel._resize_gradients` (method): Resize gradient arrays to specified dimensions for either 2D or 3D gradients.
- `CellposeModel._run_net` (method): run network on image x
- `CellposeModel._compute_masks` (method): compute masks from flows and cell probability
- `LoRALayer` (class): Defines the LoRALayer class and its behavior.
- `LoRALayer.__init__` (method): Initializes the object and sets up its internal state.
- `LoRALayer.forward` (method): Implements forward.
- `LoRALayer.merge` (method): Implements merge.
- `convert_to_lora` (function): Implements convert to lora.
- `merge_and_remove_lora` (function): Implements merge and remove lora.

## `cellpose/plot.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

- `dx_to_circ` (function): Converts the optic flow representation to a circular color representation.
- `show_segmentation` (function): Plot segmentation results (like on website).
- `mask_rgb` (function): Masks in random RGB colors.
- `mask_overlay` (function): Overlay masks on image (set image to grayscale).
- `image_to_rgb` (function): Converts image from 2 x Ly x Lx or Ly x Lx x 2 to RGB Ly x Lx x 3.
- `interesting_patch` (function): Get patch of size bsize x bsize with most masks.
- `disk` (function): Returns the pixels of a disk with a given radius and center.
- `outline_view` (function): Generates a red outline overlay onto the image.

## `cellpose/remote_config.py`

- `load_remote_config` (function): Loads remote config from disk or configuration.

## `cellpose/train.py`

- `_loss_fn_class` (function): Calculates the loss function between true labels lbl and prediction y.
- `_loss_fn_seg` (function): Calculates the loss function between true labels lbl and prediction y.
- `_reshape_norm` (function): Reshapes and normalizes the input data.
- `_get_batch` (function): Get a batch of images and labels.
- `_reshape_norm_save` (function): not currently used -- normalization happening on each batch if not load_files
- `_process_train_test` (function): Process train and test data.
- `train_seg` (function): Train the network with images for segmentation.

## `cellpose/transforms.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

- `_taper_mask` (function): Generate a taper mask.
- `unaugment_tiles` (function): Reverse test-time augmentations for averaging (includes flipping of flowsY and flowsX).
- `average_tiles` (function): Average the results of the network over tiles.
- `make_tiles` (function): Make tiles of image to run at test-time.
- `normalize99` (function): Normalize the image so that 0.0 corresponds to the 1st percentile and 1.0 corresponds to the 99th percentile.
- `normalize99_tile` (function): Compute normalization like normalize99 function but in tiles.
- `gaussian_kernel` (function): Generates a 2D Gaussian kernel.
- `smooth_sharpen_img` (function): Sharpen blurry images with surround subtraction and/or smooth noisy images.
- `move_axis` (function): move axis m_axis to first or last position
- `move_min_dim` (function): Move the minimum dimension last as channels if it is less than 10 or force is True.
- `update_axis` (function): Squeeze the axis value based on the given parameters.
- `_convert_image_3d` (function): Convert a 3D or 4D image array to have dimensions ordered as (Z, X, Y, C).
- `convert_image` (function): Converts the image to have the z-axis first, channels last.
- `normalize_img` (function): Normalize each channel of the image with optional inversion, smoothing, and sharpening.
- `resize_safe` (function): OpenCV resize function does not support uint32.
- `resize_image` (function): Resize image for computing flows / unresize for computing dynamics.
- `get_pad_yx` (function): Retrieves pad yx.
- `pad_image_ND` (function): Pad image for test-time so that its dimensions are a multiple of 16 (2D or 3D).
- `random_rotate_and_resize` (function): Augmentation by random rotation and resizing.

## `cellpose/utils.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

- `TqdmToLogger` (class): Output stream for TQDM which will output to logger module instead of the StdOut.
- `TqdmToLogger.__init__` (method): Initializes the object and sets up its internal state.
- `TqdmToLogger.write` (method): Implements write.
- `TqdmToLogger.flush` (method): Implements flush.
- `rgb_to_hsv` (function): Implements rgb to hsv.
- `hsv_to_rgb` (function): Implements hsv to rgb.
- `download_url_to_file` (function): Download object at the given URL to a local path.
- `distance_to_boundary` (function): Get the distance to the boundary of mask pixels.
- `masks_to_edges` (function): Get edges of masks as a 0-1 array.
- `remove_edge_masks` (function): Removes masks with pixels on the edge of the image.
- `masks_to_outlines` (function): Get outlines of masks as a 0-1 array.
- `outlines_list` (function): Get outlines of masks as a list to loop over for plotting.
- `outlines_list_single` (function): Get outlines of masks as a list to loop over for plotting.
- `outlines_list_multi` (function): Get outlines of masks as a list to loop over for plotting.
- `get_outline_multi` (function): Get the outline of a specific mask in a multi-mask image.
- `dilate_masks` (function): Dilate masks by n_iter pixels.
- `get_perimeter` (function): Calculate the perimeter of a set of points.
- `get_mask_compactness` (function): Calculate the compactness of masks.
- `get_mask_perimeters` (function): Calculate the perimeters of the given masks.
- `circleMask` (function): Creates an array with indices which are the radius of that x,y point.
- `get_mask_stats` (function): Calculate various statistics for the given binary masks.
- `get_masks_unet` (function): Create masks using cell probability and cell boundary.
- `stitch3D` (function): Stitch 2D masks into a 3D volume using a stitch_threshold on IOU.
- `diameters` (function): Calculate the diameters of the objects in the given masks.
- `radius_distribution` (function): Calculate the radius distribution of masks.
- `size_distribution` (function): Calculates the size distribution of masks.
- `fill_holes_and_remove_small_masks` (function): Fills holes in masks (2D/3D) and discards masks smaller than min_size.

## `cellpose/version.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.

_No top-level classes or functions found._

## `cellpose/vit_sam.py`

**Module:** Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.

- `Transformer` (class): Defines the Transformer class and its behavior.
- `Transformer.__init__` (method): Initializes the object and sets up its internal state.
- `Transformer.forward` (method): Implements forward.
- `Transformer.load_model` (method): Loads model from disk or configuration.
- `Transformer.device` (method): Get the device of the model.
- `Transformer.save_model` (method): Save the model to a file.
- `CPnetBioImageIO` (class): A subclass of the CP-SAM model compatible with the BioImage.IO Spec.
- `CPnetBioImageIO.forward` (method): Perform a forward pass of the CPnet model and return unpacked tensors.
- `CPnetBioImageIO.load_model` (method): Load the model from a file.
- `CPnetBioImageIO.load_state_dict` (method): Load the state dictionary into the model.

## `cpgrpc/client/client.py`

- `make_channel` (function): Implements make channel.
- `auth_metadata` (function): Implements auth metadata.
- `health_check` (function): Implements health check.
- `run_inference` (function): Runs inference to produce predictions from inputs.
- `upload_file` (function): Implements upload file.
- `clear_user_train_jobs` (function): Implements clear user train jobs.
- `download_file` (function): Implements download file.
- `list_files` (function): Implements list files.
- `main` (function): Entry point for running this module as a script.

## `cpgrpc/server/__init__.py`

_No top-level classes or functions found._

## `cpgrpc/server/auth.py`

- `AuthInterceptor` (class): Simple bearer token auth via metadata.
- `AuthInterceptor.__init__` (method): Initializes the object and sets up its internal state.
- `AuthInterceptor.intercept_service` (method): Implements intercept service.

## `cpgrpc/server/cellpose_remote_pb2.py`

**Module:** Generated protocol buffer code.

_Note: this module appears to be auto-generated from Protocol Buffers / gRPC._

_No top-level classes or functions found._

## `cpgrpc/server/cellpose_remote_pb2_grpc.py`

**Module:** Client and server classes corresponding to protobuf-defined services.

_Note: this module appears to be auto-generated from Protocol Buffers / gRPC._

- `HealthStub` (class): gRPC client stub for calling the remote Cellpose service.
- `HealthStub.__init__` (method): Protocol buffer-generated method __init__.
- `HealthServicer` (class): Abstract gRPC server interface for implementing the remote Cellpose service.
- `HealthServicer.Check` (method): Protocol buffer-generated method Check.
- `add_HealthServicer_to_server` (function): Registers the service implementation with a gRPC server.
- `Health` (class): Protocol buffer-generated class Health.
- `Health.Check` (method): Protocol buffer-generated method Check.
- `FileServiceStub` (class): gRPC client stub for calling the remote Cellpose service.
- `FileServiceStub.__init__` (method): Protocol buffer-generated method __init__.
- `FileServiceServicer` (class): Abstract gRPC server interface for implementing the remote Cellpose service.
- `FileServiceServicer.Upload` (method): Protocol buffer-generated method Upload.
- `FileServiceServicer.Download` (method): Protocol buffer-generated method Download.
- `add_FileServiceServicer_to_server` (function): Registers the service implementation with a gRPC server.
- `FileService` (class): Protocol buffer-generated class FileService.
- `FileService.Upload` (method): Protocol buffer-generated method Upload.
- `FileService.Download` (method): Protocol buffer-generated method Download.
- `InferenceServiceStub` (class): gRPC client stub for calling the remote Cellpose service.
- `InferenceServiceStub.__init__` (method): Protocol buffer-generated method __init__.
- `InferenceServiceServicer` (class): Abstract gRPC server interface for implementing the remote Cellpose service.
- `InferenceServiceServicer.Run` (method): Protocol buffer-generated method Run.
- `InferenceServiceServicer.ListModels` (method): Protocol buffer-generated method ListModels.
- `add_InferenceServiceServicer_to_server` (function): Registers the service implementation with a gRPC server.
- `InferenceService` (class): Protocol buffer-generated class InferenceService.
- `InferenceService.Run` (method): Protocol buffer-generated method Run.
- `InferenceService.ListModels` (method): Protocol buffer-generated method ListModels.

## `cpgrpc/server/services.py`

- `_ensure_generated` (function): Implements  ensure generated.
- `_get_quota_bytes` (function): Implements  get quota bytes.
- `_dir_size_bytes` (function): Implements  dir size bytes.
- `_ensure_quota` (function): Implements  ensure quota.
- `_sanitize_user_id` (function): Implements  sanitize user id.
- `_metadata_dict` (function): Implements  metadata dict.
- `_get_user_id` (function): Implements  get user id.
- `HealthServicer` (class): Defines the HealthServicer class and its behavior.
- `HealthServicer.Check` (method): Implements check.
- `FileService` (class): Defines the FileService class and its behavior.
- `FileService.__init__` (method): Initializes the object and sets up its internal state.
- `FileService._win_longpath` (method): Implements  win longpath.
- `FileService._user_root` (method): Implements  user root.
- `FileService._clear_user_jobs` (method): Implements  clear user jobs.
- `FileService._project_root` (method): Implements  project root.
- `FileService._safe_relpath` (method): Implements  safe relpath.
- `FileService.Upload` (method): Implements upload.
- `FileService.Download` (method): Implements download.
- `InferenceJob` (class): Defines the InferenceJob class and its behavior.
- `InferenceJob.__init__` (method): Initializes the object and sets up its internal state.
- `InferenceManager` (class): Defines the InferenceManager class and its behavior.
- `InferenceManager.__init__` (method): Initializes the object and sets up its internal state.
- `InferenceManager.submit` (method): Implements submit.
- `InferenceManager._worker` (method): Implements  worker.
- `InferenceService` (class): Defines the InferenceService class and its behavior.
- `InferenceService.__init__` (method): Initializes the object and sets up its internal state.
- `InferenceService._user_root` (method): Implements  user root.
- `InferenceService._ensure_user_path` (method): Implements  ensure user path.
- `InferenceService._resolve_item_path` (method): Implements  resolve item path.
- `InferenceService._scan_replay_items` (method): Implements  scan replay items.
- `InferenceService._scan_project_items` (method): Implements  scan project items.
- `InferenceService.ListModels` (method): Implements list models.
- `InferenceService._resolve_model_id` (method): Implements  resolve model id.
- `InferenceService.Run` (method): Implements run.
- `InferenceService._run_inference_job` (method): Implements  run inference job.
- `InferenceService._filter_masks_by_class` (method): Implements  filter masks by class.
- `InferenceService._load_classes_map` (method): Implements  load classes map.
- `InferenceService._labels_from_masks` (method): Implements  labels from masks.
- `InferenceService._load_label_item` (method): Implements  load label item.
- `InferenceService._run_training_job` (method): Implements  run training job.
- `TrainingJob` (class): Defines the TrainingJob class and its behavior.
- `TrainingJob.__init__` (method): Initializes the object and sets up its internal state.
- `TrainingManager` (class): Defines the TrainingManager class and its behavior.
- `TrainingManager.__init__` (method): Initializes the object and sets up its internal state.
- `TrainingManager.submit` (method): Implements submit.
- `TrainingManager._worker` (method): Implements  worker.
- `serve` (function): Implements serve.
