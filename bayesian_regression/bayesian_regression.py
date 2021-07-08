import matplotlib.pyplot as plt
import numpy as np


class BayesianRegression:
    def __init__(self, noise_variance=1, prior_variance=10**6):
        self.noise_variance = noise_variance
        self.prior_variance = prior_variance

        self.x_train = None
        self.t_train = None
        self.phi_train = None
        self.train_num = None
        self.feature_num = None

        self.posterior_mean = None
        self.posterior_variance = None

    @staticmethod
    def sigmoid_phi(x):
        # x_shape：Nx1
        mus = np.array([0, 2/3, 4/3])
        s = 0.1
        a = (x-mus) / s

        return 1 / (1+np.exp(-a))

    def posterior(self, x_train, t_train):
        self.x_train, self.t_train = x_train, t_train
        self.phi_train = self.sigmoid_phi(x_train)
        phi_train = self.phi_train
        self.train_num, self.feature_num = phi_train.shape

        alpha = 1 / self.prior_variance
        beta = 1 / self.noise_variance
        posterior_variance = np.linalg.inv(alpha*np.eye(self.feature_num) +
                                           beta*phi_train.T.dot(phi_train))
        posterior_mean = beta * posterior_variance.dot(phi_train.T).dot(t_train)

        self.posterior_mean = posterior_mean
        self.posterior_variance = posterior_variance

        return posterior_mean, posterior_variance

    def posterior_plot(self, title):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.x_train, self.t_train, 'bo', mfc='None')

        x_line = np.arange(0, 1.01, 0.01).reshape(-1, 1)
        phi_line = self.sigmoid_phi(x_line)
        for _ in range(5):
            sample_w = np.random.multivariate_normal(self.posterior_mean.reshape(-1),
                                                     self.posterior_variance, 1).reshape(-1, 1)
            y_line = phi_line.dot(sample_w)
            ax.plot(x_line, y_line)
        ax.set_ylim(-2, 2)
        ax.set_title(f'{title}')
        fig.savefig(f'{title}', facecolor='w')
        plt.show()

    def predict_distribution(self, phi_new):
        predict_num = phi_new.shape[0]
        predict_mean = phi_new.dot(self.posterior_mean)
        predict_variance = np.zeros(predict_num)

        for i in range(predict_num):
            phi_1row = phi_new[i, :].reshape(-1, 1)
            predict_variance[i] = phi_1row.T.dot(self.posterior_variance).dot(phi_1row)
        predict_variance += self.noise_variance

        return predict_mean.reshape(-1), predict_variance

    def predict_plot(self, title):
        x_new = np.arange(0, 1.01, 0.01).reshape(-1, 1)
        phi_new = self.sigmoid_phi(x_new)

        predict_mean, predict_variance = self.predict_distribution(phi_new)
        upper_bound = predict_mean + (predict_variance)**0.5
        lower_bound = predict_mean - (predict_variance)**0.5

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.x_train, self.t_train, "bo", mfc='None')
        ax.plot(x_new, predict_mean, color="red")
        ax.fill_between(x_new.reshape(-1), upper_bound, lower_bound, color="pink")
        ax.set_ylim(-2, 2)
        ax.set_title(f'{title}')
        fig.savefig(f'{title}', facecolor='w')
        plt.show()
