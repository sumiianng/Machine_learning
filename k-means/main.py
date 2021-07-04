import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans


# data
n = 15
x1 = np.random.randn(n, 2)
x2 = np.random.randn(n, 2) + np.array([1, 3])
x3 = np.random.randn(n, 2) + np.array([4, 4])
x = np.concatenate((x1, x2, x3), axis=0)

# show data with labels
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x[:, 0], x[:, 1], 'o')
ax.set_title("Data with labels")
fig.savefig("Data_with_labels")
plt.show()

# show data without labels
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x1[:, 0], x1[:, 1], 'o')
ax.plot(x2[:, 0], x2[:, 1], 'o')
ax.plot(x3[:, 0], x3[:, 1], 'o')
ax.set_title("Data without labels")
fig.savefig("Data_without_labels")
plt.show()

# train
k = 3
km = KMeans(k)
t = km.train(x)

# show reuslts
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(k):
    ax.plot(x[t==i, 0], x[t==i, 1], 'o')
ax.set_title("Results")
fig.savefig("Results")
plt.show()