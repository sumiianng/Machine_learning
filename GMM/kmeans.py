import numpy as np

class KMeans:
    def __init__(self, k):
        self.k = k
        
        self.image = None
        self.img_arr = None
        self.H = None
        self.W = None
        self.color_num = None
        
        self.x = None
        self.mu = None
        self.t = None  # N(H*W)*K
    
    def image_initial(self, image):
        self.image = image
        self.img_arr = np.array(image) / 255
        self.H, self.W = self.img_arr[:, :, 0].shape
        
        if self.img_arr.ndim == 3:
            self.color_num = 3
        else:
            self.color_num = 1
            
        self.x = self.img_arr.reshape(-1,  self.color_num)
        self.mu = np.zeros((self.k, self.x.shape[1]))
    
    def cost(self):
        return np.sum(self.x - self.mu[self.t, :])
    
    def e_step(self):
        for i in range(self.k):
            self.mu[i, :] = np.sum(self.x[self.t == i, :], axis=0) / (np.sum(self.t == i)+10**(-7))
    
    def m_step(self):
        distance = np.zeros((self.k, self.H*self.W))
        for i in range(self.k):
            distance[i, :] = np.sum((self.x - self.mu[i, :])**2, axis=1)
        
        self.t = np.argmin(distance, axis=0)
        
    def train(self, image):
        self.image_initial(image)
        self.m_step()
        mu_old = self.mu.copy()
        epoch = 0
        while True:
            epoch += 1
            self.e_step()
            self.m_step()
            
            error = np.sum((self.mu - mu_old)**2)
            if (epoch % 1) == 0:
                print(f"Epoch: {epoch:>2}   Error: {error:>.2f}")
            if  error < 0.01:
                break
            mu_old = self.mu.copy()
            
        return self.mu
