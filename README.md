# GIS Applications
Python Scripts for GIS

# Python Packages
- numpy
- scipy
- scikit-image
- pillow

# Finding ridges and valleys
1. Perform Sobel convolution on DEM to prepare slope maps (vertical and horizontal separately)
2. Generate points that follow slope up or down on the DEM
3. Aggregate traces of the crawling points

# To-do
- Normalize preview arrays to [0, 255] before saving
- Normalize slope maps to [0., 1.]
