import matplotlib.pyplot as plt
from PIL import Image
from kmeans import KMeans
from GMM import GMM

img = Image.open('data/scenery.jpg')

ks = [3, 5, 7, 10]
for k in ks:
    print(f"========== k = {k} ==============")
    print("========== K_means start =========")
    # Find the initial solution of mu for GMM.
    km = KMeans(k)
    mu = km.train(img)

    print("========== GMM start =========")
    gmm = GMM(k, mu)
    gmm.train(img)

    # learning curve
    gmm.learning_curve()

    # results
    gmm.img_split()

# all results
img_original = Image.open('data/scenery.jpg')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img_original)
ax.axis('off')
plt.show()

ks = [3, 5, 7, 10]
fig = plt.figure()
for i, k in enumerate(ks):
    img = Image.open(f'results/image_segmentation_k={k}.png')
    ax = plt.subplot(2, 2, i+1)
    ax.imshow(img)
    ax.axis('off')

plt.subplots_adjust(wspace=-0.1, hspace=0.03)
plt.show()
