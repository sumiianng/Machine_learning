# Bayesian Regression
In the project, we generate the data from the sin function and add the noises. The we use the bayesian regression with sigmoid basis function to fit the nonlinear data. Then, we show the samples of posterior distribution and show the prediction fucntion. The comparison of the results between different number of training data would be showed.  

&emsp;
## Data
We generate the 60 data from the sin function and add some noises, so we get the data and targets we have to use later.  

&emsp;
## Posterior and Prediction Distribution
We use the data to train the model and we can get the posterior distribution of parameters. We sample from the posterior distribution and show the results on the plot. Then, we calculate the prediction distribution, and show the mean and the region between one deviation. Finally, compare the results of different training sample number.  

* **Hyper Parameters**
1. noise_variance = 10**0   (1/beta)
2. prior_variance = 10**6   (1/alpha)

* **Number of Sampleling**： [5, 10, 30, 60]

&emsp;
* **Posterior Distribution**  
&emsp;
<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/bayesian_regression/posterior_distribution_5.png" height="200px">
<img src="https://github.com/sumiianng/image_storage/blob/main/bayesian_regression/posterior_distribution_10.png" height="200px">
</div>

<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/bayesian_regression/posterior_distribution_30.png" height="200px">
<img src="https://github.com/sumiianng/image_storage/blob/main/bayesian_regression/posterior_distribution_60.png" height="200px">
</div>

&emsp;
* **Prediction Distribution**  
&emsp;
<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/bayesian_regression/prediction_distribution_5.png" height="200px">
<img src="https://github.com/sumiianng/image_storage/blob/main/bayesian_regression/prediction_distribution_10.png" height="200px">
</div>

<div align="center">
<img src="https://github.com/sumiianng/image_storage/blob/main/bayesian_regression/prediction_distribution_30.png" height="200px">
<img src="https://github.com/sumiianng/image_storage/blob/main/bayesian_regression/prediction_distribution_60.png" height="200px">
</div>



