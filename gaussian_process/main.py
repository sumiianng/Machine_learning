import scipy.io as io
from gaussian_process import GP

# data
data = io.loadmat('gp.mat')
x = data['x']
t = data['t']

train_num = 60
x_train = x[:train_num, :]
x_test = x[train_num:, :]
t_train = t[:train_num]
t_test = t[train_num:]

# train and predict results
theta_list = [[0, 0, 0, 1],
              [1, 4, 0, 0],
              [1, 4, 0, 5],
              [1, 32, 5, 5]]

for i, thetas in enumerate(theta_list):
    gp = GP(thetas=thetas)
    gp.train(x_train, t_train)
    gp.plot(x_test, t_test, title=f"theta_{i}")
    train_error, test_error = gp.rms_error()

    print(f'train error: {train_error:>.2f}  test error: {test_error:>.3f}')

# ARD
gp.ard()
gp.plot(x_test, t_test, "ARD")
train_error, test_error = gp.rms_error()
print(f'train error: {train_error:>.2f}  test error: {test_error:>.3f}')
