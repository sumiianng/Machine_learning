import numpy as np


def generate_data(data_num=50, noise_variance=0.01):
    x = np.random.rand(data_num)
    noise = np.random.randn(data_num) * noise_variance**0.5
    t = (np.sin((x+0.1)*1.4*np.pi) + noise).reshape(-1, 1)

    return x.reshape(-1, 1), t
