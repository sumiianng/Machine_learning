import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class GMM:
    def __init__(self, k, mu=None):
        self.k = k

        self.image = None
        self.img_arr = None
        self.H = None
        self.W = None
        self.n = None
        self.color_num = None

        self.x = None
        self.t = None  # N(H*W)*K
        self.r = None
        self.r_u = None
        self.log_lhs = []

        if mu is None:
            self.mu = np.random.rand(k, self.color_num)
        else:
            self.mu = mu

        self.pi = np.ones(self.k) / self.k
        self.cov = None

        self.adjust = False

    def image_initial(self, image):
        self.image = image
        self.img_arr = np.array(image) / 255
        self.H, self.W = self.img_arr[:, :, 0].shape
        self.n = self.H + self.W

        if self.img_arr.ndim == 3:
            self.color_num = 3
        else:
            self.color_num = 1

        self.x = self.img_arr.reshape(-1,  self.color_num)
        self.cov = np.array([np.eye(self.color_num)]*self.k) * 0.01

    def gaussian_value(self, x, mu, cov):
        p = len(mu)
        n = x.shape[0]
        es = np.zeros(n)
        if np.linalg.det(cov) == 0:
            cov = cov + 0.0001*np.identity(cov.shape[0])
            adjust = True
        else:
            adjust = False

        coe = (2*np.pi)**(-p/2) * np.linalg.det(cov)**-0.5
        temp1 = x - mu
        temp2 = temp1.dot(np.linalg.inv(cov))

        for i in range(n):
            es[i] = (temp2[i].dot(temp1[i].reshape(-1, 1))) * -0.5

        return coe * np.exp(es), adjust

    def cal_log_likelihood(self):
        return np.sum(np.log(np.sum(self.r_u, axis=1)))

    def e_step(self):
        x, k = self.x, self.k

        gaussian_values = np.zeros((x.shape[0], k))
        adjusts = np.zeros(k)

        for i in range(k):
            gaussian_values[:, i], adjusts[i] = self.gaussian_value(self.x, self.mu[i], self.cov[i])

        self.r_u = gaussian_values * self.pi
        self.r = gaussian_values*self.pi / np.sum(gaussian_values*self.pi, axis=1).reshape(-1, 1)
        self.t = np.argmax(self.r, axis=1)

        self.adjust = adjusts.any()

    def m_step(self, first=None):
        x, r, k = self.x, self.r, self.k
        N = np.sum(r, axis=0)
        # print(N)
        self.pi = N / self.n
        for i in range(k):
            if not first:
                self.mu[i] = np.sum(x * r[:, i].reshape(-1, 1), axis=0) / (N[i]+10**-7)
            var = 0
            for j in range(x.shape[0]):
                vec = (x[j]-self.mu[i]).reshape(-1, 1)
                var += r[j, i]*vec.dot(vec.T)
            self.cov[i] = var / (N[i]+10**-7)

        if self.adjust:
            self.log_lhs[-1] = self.cal_log_likelihood()
        else:
            self.log_lhs.append(self.cal_log_likelihood())

    def train(self, image):
        self.image_initial(image)
        epoch = 0
        self.e_step()
        self.m_step(first=True)

        print(f"Epoch: {epoch:>2}")
        while True:
            epoch += 1

            self.e_step()
            self.m_step()
            if self.adjust:
                epoch -= 1

            error = np.abs(self.log_lhs[-1] - self.log_lhs[-2])

            if (epoch % 1) == 0:
                print(f"Epoch: {epoch:>2}   Error: {error:>.2f}   log_lh: {self.log_lhs[-1]:>.2f}")

            if (error < 0.1) | (epoch > 100):
                break

    def img_split(self):
        x, k, t = self.x, self.k, self.t
        x_s = np.zeros_like(x)
        for i in range(k):
            x_s[t == i, :] = self.mu[i]

        img_s_arr = x_s.reshape(self.H, self.W, -1) * 255
        img_s_arr = img_s_arr.astype(np.uint8)
        img_s = Image.fromarray(img_s_arr)
        img_s.save(f'image_segmentation_k={self.k}.png')
        img_s.show()

    def learning_curve(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = np.arange(1, len(self.log_lhs)+1)
        y = self.log_lhs
        ax.plot(x, y)
        ax.set_title(f"log_likelihood k={self.k}")
        fig.savefig(f"log_likelihood k={self.k}", facecolor='w')
        plt.show()
