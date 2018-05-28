# GIS Applications
Python Scripts for GIS

# Python Packages
- numpy
- scipy
- scikit-image
- pillow
- tensorflow

# Definitions
- climber: A dot crawling in DEM upward or downward
- crawl: An iteration to move a climber's coordinate and trace its previous coordinate in the footprint map
- footprint map: A 2D array to keep track of climbers' footprint
- trace: The act to change values in foorprint

# Finding Ridges and Valleys with DEM
1. Perform Sobel convolution on DEM to prepare slope maps (vertical and horizontal separately)
2. Generate points that follow slope up or down on the DEM (random or evenly distributed?)
3. Aggregate traces of the crawling points in a new map
4. Apply threshold on the trace map (local threshold or global?)

# Issues and Improvements Yet to be Implemented
- Use tensorflow.image.resize_images to downsample numpy.memmap arrays
- Early stop crawling if the climber is lingering -> What algorithm detects lingering fast?
- Change slope vector to uint16 and use Bresenham's algorithm for calculating new coordinate of climbers
- Replace float array with integer whenever possible

memmap benchamrk
Assigning uint16 array: 0.01600337028503418
Adding uint16 array: 0.008001565933227539
Multiplying uint16 array: 0.010001897811889648
Resizing uint16 array: 32.55651021003723
Assigning flaot16 array: 0.013002395629882812
Adding flaot16 array: 0.20104002952575684
Multiplying flaot16 array: 0.1930387020111084
Resizing flaot16 array: 73.62072110176086
