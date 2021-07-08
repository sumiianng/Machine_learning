from data_generator import generate_data
from bayesian_regression import BayesianRegression


train_num = 60
x_train_all, t_train_all = generate_data(data_num=train_num)

noise_variance = 10**0
prior_variance = 10**6

sample_num_list = [5, 10, 30, 60]
for sample_num in sample_num_list:
    x_train = x_train_all[:sample_num]
    t_train = t_train_all[:sample_num]

    bayesian_regression = BayesianRegression(noise_variance, prior_variance)
    bayesian_regression.posterior(x_train, t_train)

    bayesian_regression.posterior_plot(f'posterior_distribution_{sample_num}')
    bayesian_regression.predict_plot(f'prediction_distribution_{sample_num}')
