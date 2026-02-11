import os
import shutil
import glob
from PyQt6.QtCore import pyqtSignal
from guv_app.workers.base_worker import BaseWorker

class PromoteWorker(BaseWorker):
    """
    Worker for promoting prediction files (_pred.npy) to segmentation files (_seg.npy).
    """
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self._is_running = True

    def run(self):
        self.progress.emit("Starting promotion of predictions...")
        # Find all _pred.npy files in the folder
        search_path = os.path.join(self.folder_path, "*_pred.npy")
        pred_files = glob.glob(search_path)
        
        total = len(pred_files)
        if total == 0:
            self.progress.emit("No _pred.npy files found in the selected folder.")
            self.finished.emit()
            return

        promoted = 0
        skipped = 0
        for i, pred_path in enumerate(pred_files):
            if not self._is_running:
                break
            
            dir_name, file_name = os.path.split(pred_path)
            new_file_name = file_name.replace("_pred.npy", "_seg.npy")
            seg_path = os.path.join(dir_name, new_file_name)
            
            try:
                if os.path.exists(seg_path):
                    try:
                        same_size = os.path.getsize(seg_path) == os.path.getsize(pred_path)
                        seg_newer = os.path.getmtime(seg_path) >= os.path.getmtime(pred_path)
                        if same_size and seg_newer:
                            skipped += 1
                            self.progress.emit(f"Skipped {file_name} (already promoted) ({i+1}/{total})")
                            continue
                    except OSError:
                        # If stat fails, fall back to overwrite.
                        pass
                    try:
                        os.remove(seg_path)
                    except OSError:
                        pass

                # Prefer hardlink on same volume for O(1) promotion.
                try:
                    os.link(pred_path, seg_path)
                except OSError:
                    # Fall back to a fast copy without metadata.
                    shutil.copyfile(pred_path, seg_path)

                promoted += 1
                self.progress.emit(f"Promoted {file_name} ({i+1}/{total})")
            except Exception as e:
                self.progress.emit(f"Failed to promote {file_name}: {e}")
        
        self.progress.emit(f"Promotion complete. Promoted {promoted}, skipped {skipped}.")
        self.finished.emit()

    def stop(self):
        self._is_running = False
