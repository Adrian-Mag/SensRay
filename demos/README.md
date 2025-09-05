# SeisRay Demonstrations

This folder contains Jupyter notebook demonstrations showing various applications of the `seisray` package for seismic ray tracing and travel time calculations.

## Notebooks Overview

### 01_basic_travel_times.ipynb
**Basic Travel Time Calculations**
- Calculate P and S wave travel times for different distances and source depths
- Compare different Earth models (iasp91, prem, ak135)
- Create travel time curves and time-distance diagrams
- Understand the effects of source depth on travel times

**Key Learning Points:**
- How to use `TravelTimeCalculator` and `EarthModelManager`
- Basic seismic wave propagation concepts
- Model comparison and uncertainty assessment

### 02_ray_path_visualization.ipynb
**Ray Path Visualization**
- Extract and visualize seismic ray paths through 1D Earth models
- Create circular Earth cross-sections with ray paths
- Analyze ray path properties (turning points, maximum depths)
- Compare P and S wave ray geometries

**Key Learning Points:**
- How to use `RayPathTracer` and `EarthPlotter`
- Understanding ray path curvature and geometry
- Visualization techniques for seismic ray propagation

### 03_earth_model_comparison.ipynb
**Earth Model Comparison**
- Compare velocity structures of different 1D Earth models
- Analyze differences in travel times between models
- Visualize velocity profiles and discontinuities
- Understand the impact of model choice on seismic analysis

**Key Learning Points:**
- Detailed Earth model analysis
- Velocity structure interpretation
- Statistical comparison of model predictions
- Resolution and uncertainty in Earth models

### 04_sensitivity_kernels.ipynb
**Sensitivity Kernels and Tomography**
- Compute and visualize sensitivity kernels for seismic tomography
- Understand kernel stacking and coverage analysis
- Perform resolution testing with synthetic data
- Explore trade-offs between resolution and data coverage

**Key Learning Points:**
- How to use `SensitivityKernel` class
- Understanding tomographic resolution
- Kernel-based inversion concepts
- Quality assessment for tomographic studies

### 05_practical_applications.ipynb
**Practical Seismology Applications**
- Earthquake location and magnitude estimation
- Regional velocity structure studies
- Station array design and optimization
- Quality control for seismic data

**Key Learning Points:**
- Real-world applications of ray tracing
- Network design principles
- Data quality assessment techniques
- Practical problem-solving in seismology

## Getting Started

1. **Prerequisites**: Make sure you have the seisray package installed and configured:
   ```bash
   cd /path/to/masters_project
   pip install -e .
   ```

2. **Required Dependencies**:
   - numpy
   - matplotlib
   - scipy
   - obspy
   - jupyter (for running notebooks)

3. **Running the Notebooks**: Start Jupyter and navigate to the demos folder:
   ```bash
   jupyter notebook
   ```

## Notebook Structure

Each notebook follows a similar structure:
- **Introduction**: Learning objectives and overview
- **Setup**: Import statements and basic configuration
- **Core Concepts**: Step-by-step demonstrations
- **Analysis**: Detailed parameter studies
- **Visualization**: Comprehensive plots and figures
- **Summary**: Key findings and practical insights

## Tips for Users

### For Beginners:
- Start with `01_basic_travel_times.ipynb` to understand fundamentals
- Run cells sequentially to build understanding
- Experiment with parameters to see how results change
- Read the summary sections for key takeaways

### For Advanced Users:
- Focus on `04_sensitivity_kernels.ipynb` and `05_practical_applications.ipynb`
- Modify examples for your specific research problems
- Use the code as templates for your own analysis
- Explore the trade-offs and limitations discussed

### For Developers:
- Study the implementation patterns for extending seisray
- Note the error handling and validation approaches
- Consider the visualization techniques for your own tools
- Use the notebooks as integration tests for new features

## Educational Value

These notebooks are designed for:
- **Students**: Learning seismological concepts and computational methods
- **Researchers**: Understanding tool capabilities and limitations
- **Instructors**: Teaching materials for seismology courses
- **Practitioners**: Reference implementations for common problems
