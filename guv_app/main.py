import sys
import logging
import os
import argparse
from pathlib import Path
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon
except ImportError:
    print("PyQt6 not installed. Please install it with 'pip install PyQt6'")
    sys.exit(1)

from guv_app.controllers.home_controller import HomeController
from guv_app.views.home_view import HomeView
from guv_app.views.style import DarkPalette, stylesheet
import pyqtgraph as pg
from cellpose import train as cellpose_train


def _resolve_app_icon() -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "MultiCellPose.png",
        repo_root / "assets" / "icons" / "MultiCellPose.png",
        repo_root / "cellpose" / "logo" / "cellpose.ico",
        repo_root / "cellpose" / "logo" / "logo.png",
        repo_root / "t.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _set_windows_app_id(app_id: str) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def main():
    """
    Main function to initialize and run the GUVpose application.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout  # Log to standard output
    )

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-debug", action="store_true", default=False)
    parser.add_argument("--train-debug-steps", type=int, default=3)
    cli_args, qt_args = parser.parse_known_args(sys.argv[1:])
    cellpose_train.set_train_debug(cli_args.train_debug, cli_args.train_debug_steps)

    _set_windows_app_id("OpenBiomolecularFluidics.MultiCellPose")
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("MultiCellPose")
    app.setOrganizationName("Open Biomolecular Fluidics")
    app.setOrganizationDomain("openbiofluidics")

    icon_path = _resolve_app_icon()
    if icon_path is not None:
        app_icon = QIcon(str(icon_path))
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)

    pg.setConfigOptions(imageAxisOrder="row-major")
    app.setStyle("Fusion")
    app.setPalette(DarkPalette())
    app.setStyleSheet(stylesheet())

    # 1. Initialize Home View
    view = HomeView()

    # 2. Initialize Controller (wires everything together)
    controller = HomeController(view, app)
    controller.connect_signals()

    # Show the main window
    view.show()

    # Start the application event loop
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
