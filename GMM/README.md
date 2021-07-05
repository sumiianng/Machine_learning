# Gaussian Mixture Model
In the project, we use the gaussian mixture model to do the image segmentation problem.The goal of segmentation is to partition an image into many regions. We use k-means algorithm to find the initial solution of parameter mu. Then, update the parameters and the cluster of each pixels by EM algorithm. Finally, we show the results and compare the results for different k-value setting.

## Data
Find an image for image segmentation.  

<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/GMM/scenery.jpg" height="200px">
</div>

## Training
Show the learning curve for the setting of different value of k.  


<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/GMM/log_likelihood%20k%3D3.png" height="200px">
<img src="https://github.com/sumiianng/image_storage/blob/main/GMM/log_likelihood%20k%3D5.png" height="200px">
</div>

<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/GMM/log_likelihood%20k%3D7.png" height="200px" >
<img src="https://github.com/sumiianng/image_storage/blob/main/GMM/log_likelihood%20k%3D10.png" height="200px" >
</div>

## Results
Show the image segmentation results for the setting of different value of k.  

The original image  
<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/GMM/scenery.jpg" height="200px" >
</div>

The results of image segmentation, and the values of k from upper left to lower right are 3, 5, 7, 10.
<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/GMM/image_segmentation_k%3D3.png" height="200px" >
<img src="https://github.com/sumiianng/image_storage/blob/main/GMM/image_segmentation_k%3D5.png" height="200px" >
</div>

<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/GMM/image_segmentation_k%3D7.png" height="200px" >
<img src="https://github.com/sumiianng/image_storage/blob/main/GMM/image_segmentation_k%3D10.png" height="200px" >
</div>
