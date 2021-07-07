# Gaussian Process for Regression
In the project, we implement the gaussin process (GP) for regression. We split the data into train set and test set. Then, the predict distribution of the new data would be visualized. Otherwise, in the gaussian process, we use differen kernel by setting the different value of theta. Finally, we compare the results of different kernels, and use automatic relevance determination (ARD) to select the value of parameters. 

&emsp;
## Data
The data 'gp.mat' contain 100 data and 100 targets. We use the first 60 as the training set, and the last 40 as the testing set.

&emsp;
## Training and Predicting Results
We use different value of theta to represent different kernels. Then, we train the model and plot the predict results. Otherwise, calculate the root-mean-square error of the training set and the testing set. We can compare the results of different value of theta.  

* **Theta combinations**
1. linear kernel：[0, 0, 0, 1]
2. squared exponential kernel： [1, 4, 0, 0]
3. exponential-quadratic kernel ： [1, 4, 0, 5]
4. exponential-quadratic kernel： [1, 32, 5, 5]

* **Plot**  
&emsp;
<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/gaussian_process/theta_0.png" height="200px">
<img src="https://github.com/sumiianng/image_storage/blob/main/gaussian_process/theta_1.png" height="200px">
</div>

<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/gaussian_process/theta_2.png" height="200px">
<img src="https://github.com/sumiianng/image_storage/blob/main/gaussian_process/theta_3.png" height="200px">
</div>

&emsp;
* **Root-Mean-Square Error**  
&emsp;

Theta    |  0, 0, 0, 1   |  1, 4, 0, 0   |  1, 4, 0, 5   |  1, 32, 5, 5 
---------|:-------------:|--------------:|--------------:|--------------
Train    |     6.658     |     1.052     |     1.029     |     0.964     
Test     |     6.749     |     1.299     |     1.286     |     1.258     


&emsp;
## Automatic Relevance Detemination
Generally, we have to set the the value of hyper-parameters by ourselves. However, we can use ARD to set the value of theta automatically. 

* **Plot**  
&emsp;
<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/gaussian_process/ARD.png" height="200px">
</div>

&emsp;
* **Root-Mean-Square Error**  
Train：0.968  
Test ：1.240

