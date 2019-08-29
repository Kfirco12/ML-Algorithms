Here I implemented the k-means algorithm for image compression, i.e. I
implemented the k-means algorithm on the image pixels and then replace each pixel by its centroid.

Originally, the initial centroids in k-means are randomly generated. For
reproducibility purposes, the centroid initialization provided in a python script named "init centroids.py".

Input:
The k-means script use the picture 'dog.jpeg', and it should be in the same directory of the k_mean.py script.

Output:
The output of the 'k_meean.py' script will be the new centroids with k=[2, 4, 8, 16] for 10 iterations
of the provided picture.