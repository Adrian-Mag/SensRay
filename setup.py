"""
Setup script for the seisray package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="seisray",
    version="0.2.0",
    author="PhD Student",
    description="Sensitivity kernels and 3D visualization for seismic tomography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
        "obspy>=1.3.0",
        "pyvista>=0.40.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            "mypy",
        ],
        "notebooks": [
            "jupyter",
            "ipython",
        ],
        "geographic": [
            "cartopy>=0.20.0",
        ],
    },
    keywords="seismology, tomography, kernels, visualization",
    project_urls={
        "Bug Reports": "https://github.com/username/seisray/issues",
        "Source": "https://github.com/username/seisray",
    },
)
