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
