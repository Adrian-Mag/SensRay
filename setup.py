"""
Setup script for the seisray package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sensray",
    version="0.3.0",
    author="PhD Student",
    description="Seismic ray tracing, travel times, and 3D visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        'sensray': ['models/*.nd'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
        "obspy>=1.3.0",
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
            "ipywidgets",
        ],
        "geographic": [
            "cartopy>=0.20.0",
        ],
        "meshing": [
            "pyvista>=0.40.0",
            "meshio>=5.0.0",
            "pygmsh>=7.1.17",
        ],
        "all": [
            "pytest>=6.0",
            "black",
            "flake8",
            "mypy",
            "jupyter",
            "ipython",
            "ipywidgets",
            "cartopy>=0.20.0",
            "pyvista>=0.40.0",
            "meshio>=5.0.0",
            "pygmsh>=7.1.17",
        ],
    },
    keywords="seismology, ray-tracing, travel-times, earth-models",
    project_urls={
        "Bug Reports": "https://github.com/username/sensray/issues",
        "Source": "https://github.com/username/sensray",
        "Documentation": "https://github.com/username/sensray/tree/main/demos",
    },
)
