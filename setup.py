import setuptools
from setuptools import setup

install_deps = [
    'numpy>=1.20.0',
    'scipy',
    'protobuf',
    'grpcio',
    'natsort',
    'tifffile',
    'tqdm',
    'torch>=1.6',
    'torchvision',
    'opencv-python-headless',
    'fastremap',
    'imagecodecs',
    'roifile',
    'fill-voids',
    'segment_anything',
    'pandas',
    'nd2',
    'pynrrd',
    'readlif'
]

image_deps = ['nd2', 'pynrrd', 'readlif']

gui_deps = [
    'pyqtgraph>=0.12.4', "pyqt6", "pyqt6.sip", 'qtpy', 'superqt',
    'nd2', 'pynrrd', 'readlif',
]

docs_deps = [
    'sphinx>=3.0',
    'sphinxcontrib-apidoc',
    'sphinx_rtd_theme',
    'sphinx-argparse',
]

distributed_deps = [
    'dask',
    'distributed',
    'dask_image',
    'pyyaml',
    'zarr',
    'dask_jobqueue',
    'bokeh',
    'pyarrow',
]

bioimageio_deps = [
    'bioimageio.core',
]

try:
    import torch
    a = torch.ones(2, 3)
    from importlib.metadata import version
    ver = version("torch")
    major_version, minor_version, _ = ver.split(".")
    if major_version == "2" or int(minor_version) >= 6:
        install_deps.remove("torch>=1.6")
except:
    pass

try:
    import PyQt6
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except:
    pass

try:
    import PySide2
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except:
    pass

try:
    import PySide6
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except:
    pass

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except Exception:
    try:
        # fallback for Windows legacy encodings
        with open("README.md", "r", encoding="latin-1", errors="ignore") as fh:
            long_description = fh.read()
    except Exception:
        # installer builds can omit README.md; keep setup working without it
        long_description = "anatomical segmentation algorithm"

setup(
    name="multicellpose", license="BSD-3-Clause", author="Marcus Fletcher",
    author_email="mfletch1@users.noreply.github.com",
    description="MultiCellPose: segmentation and analysis toolkit for multimodal microscopy", long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrcsfltchr/MultiCellPose", setup_requires=[
        'pytest-runner',
        'setuptools_scm',
    ], packages=setuptools.find_packages(), use_scm_version={
        # allow installs from ZIPs/without .git by falling back to a static version
        "fallback_version": "0.0.0",
        # PyPI/TestPyPI reject local version segments like +g<sha>.d<date>
        "local_scheme": "no-local-version",
    },
    py_modules=["train_headless", "run_server"],
    install_requires=install_deps, tests_require=['pytest'], extras_require={
        'docs': docs_deps,
        'gui': gui_deps,
        'distributed': distributed_deps,
        'bioimageio': bioimageio_deps,
        'all': gui_deps + distributed_deps + image_deps + bioimageio_deps,
    }, include_package_data=True, classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ], entry_points={'console_scripts': [
        'multicellpose-gui = guv_app.main:main',
        'multicellpose-server = run_server:main',
        'multicellpose-train = train_headless:main',
    ]})
