import matplotlib.pyplot as plt
import numpy as np


class GP:
    def __init__(self, beta=1, thetas=[1, 32, 5, 5]):
        self.beta = beta
        self.thetas = thetas
        self.x = None
        self.t = None
        self.x_test = None
        self.t_test = None
        self.cov = None

    @staticmethod
    def kernel(x1, x2, thetas):
        x1 = np.array(x1)
        x2 = np.array(x2)

        result = np.zeros((x1.shape[0], x2.shape[0]))

        if thetas[0] != 0:
            if thetas[1] != 0:
                for i in range(x1.shape[0]):
                    result[i, :] = np.sum((x1[i, :]-x2) ** 2, axis=1)
                result = np.exp(-thetas[1] * result / 2)
            else:
                result = np.ones((x1.shape[0], x2.shape[0]))

        result += thetas[2] + thetas[3]*x1.dot(x2.T)

        return result

    def train(self, x, t):
        self.x = x
        self.t = t
        thetas = self.thetas

        cov = GP.kernel(x, x, thetas) + np.eye(x.shape[0])/self.beta
        self.cov = cov

    def predict(self, x_new):
        x, t, beta, thetas = self.x, self.t, self.beta, self.thetas
        k = GP.kernel(x, x_new, thetas)
        CN = GP.kernel(x, x, thetas) + np.eye(x.shape[0])/self.beta
        CN_inv = np.linalg.inv(CN)

        mu = (k.T.dot(CN_inv).dot(t)).reshape(-1)
        var = np.zeros(x_new.shape[0])
        for i in range(len(var)):
            xi = x_new[i, :].reshape(1, -1)
            ki = k[:, i].reshape(-1, 1)
            c = GP.kernel(xi, xi, thetas) + 1/beta

            var[i] = c - ki.T.dot(CN_inv).dot(ki)

        return mu, var

    def plot(self, x_test, t_test, title):
        self.x_test = x_test
        self.t_test = t_test
        # sp = np.floor(np.max(x_test))
        # ep = np.ceil(np.min(x_test))
        sp = 0
        ep = 2

        xs = np.arange(sp, ep+0.01, 0.01).reshape(-1, 1)

        mu, var = self.predict(xs)
        upper = mu + var**0.5
        lower = mu - var**0.5

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_test, t_test, 'bo', mfc='None')
        ax.plot(xs, mu, color='red')
        ax.fill_between(xs.reshape(-1), upper, lower, color='pink')
        ax.set_title(f'{self.thetas}')
        fig.savefig(f'{title}', facecolor='w')
        plt.show()

    def rms_error(self):
        x, t = self.x, self.t
        x_test, t_test = self.x_test, self.t_test

        mu, var = self.predict(x)
        rms_error_train = (np.sum((t.reshape(-1)-mu)**2)/x.shape[0]) ** 0.5

        mu_test, var_test = self.predict(x_test)
        rms_error_test = (np.sum((t_test.reshape(-1)-mu_test)**2)/x_test.shape[0]) ** 0.5

        return rms_error_train, rms_error_test

    def ard(self, eta=0.001):
        x, t = self.x, self.t

        thetas = np.ones(4)
        gradients = np.zeros(4)

        CN = GP.kernel(x, x, thetas) + np.eye(x.shape[0])/self.beta
        CN_inv = np.linalg.inv(CN)
        log_lh = (np.log(np.linalg.det(CN))/-2 - t.T.dot(CN_inv).dot(t)/2)[0, 0]

        epoch = 0
        while True:
            epoch += 1
            for i in [1, 2, 3]:
                temp1 = np.trace(CN_inv.dot(self.diff_CN(i, thetas))) / -2
                temp2 = t.T.dot(CN_inv).dot(self.diff_CN(i, thetas)).dot(CN_inv).dot(t) / 2
                gradients[i] = temp1 + temp2

            log_lh_old = log_lh

            thetas += eta * gradients
            CN = GP.kernel(x, x, thetas) + np.eye(x.shape[0])/self.beta
            CN_inv = np.linalg.inv(CN)

            log_lh = (np.log(np.linalg.det(CN))/-2 - t.T.dot(CN_inv).dot(t)/2)[0, 0]

            error = log_lh-log_lh_old
            # print(f'epoch:{epoch:>3}  error:{error:>.3f}')
            if  np.abs(error) <= 0.001:
                break

            self.thetas = np.round(thetas, 2)

    def diff_CN(self, k, thetas):
        x = self.x
        n = x.shape[0]

        if k == 0:
            results = np.zeros((n, n))
            for i in range(n):
                results[i, :] = np.sum((x[i, :]-x) ** 2, axis=1)
            results = np.exp(-thetas[1] * results / 2)

            return results

        elif k == 1:
            results = np.zeros((n, n))
            for i in range(n):
                results[i, :] = np.sum((x[i, :]-x) ** 2, axis=1)
            results = np.exp(results/-2) * thetas[0]

            return results

        elif k == 2:
            return np.ones((n, n))
        else:
            return x.dot(x.T)
