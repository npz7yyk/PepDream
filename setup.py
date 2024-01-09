import sys

# limit python version to 3.10 or later
if sys.version_info < (3, 10):
    print("PepDream requires Python 3.10 or later, please upgrade your Python version")
    exit(1)

# Check if Cython is available or if it needs to be installed
try:
    from Cython.Build import cythonize
except ImportError:
    print("Cython needed for installation, please install it manually")
    exit(1)

# Check if setuptools is available or if it needs to be installed
try:
    from setuptools import setup, find_packages
except ImportError:
    print("setuptools needed for installation, please install it manually")
    exit(1)

DISTNAME = "PepDream"
VERSION = "0.1.0"
DESCRIPTION = "PepDream: automatic feature extraction for peptide identification from tandem mass spectra"
AUTHOR = "Yikang Yue, University of Science and Technology of China (USTC)"
AUTHOR_EMAIL = "yyk2020@mail.ustc.edu.cn"
URL = "https://github.com/npz7yyk/PepDream"
LICENSE = "MIT"

CLASSIFIERS = ["Natural Language :: English",
               "Development Status :: 3 - Alpha",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Topic :: Scientific/Engineering :: Bio-Informatics",
               "Operating System :: MacOS",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               "Programming Language :: Python :: 3 :: Only"]


def main():
    setup(
        name=DISTNAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        packages=find_packages(include=["pepdream"]),
        url=URL,
        platforms=["any"],
        install_requires=[
            "h5py",
            "tqdm",
            "numpy",
            "scikit-learn",
            "torch",
            "Cython"
        ],
        classifiers=CLASSIFIERS,
        ext_modules=cythonize(
            "pepdream/qvalue_acc.pyx",
            build_dir="build"
        ),
        entry_points={
            "console_scripts": [
                "pepdream = pepdream.main:main"
            ]
        }
    )


if __name__ == "__main__":
    main()
