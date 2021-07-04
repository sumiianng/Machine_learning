import numpy as np

class KMeans:
    def __init__(self, k):
        self.k = k
            
        self.x = None
        self.mu = None
        self.t = None  
    
    def cost(self):
        return np.sum(self.x - self.mu[self.t, :])
    
    def e_step(self):
        for i in range(self.k):
            self.mu[i, :] = np.sum(self.x[self.t == i, :], axis=0) / (np.sum(self.t == i)+10**(-7))
    
    def m_step(self):
        distance = np.zeros((self.k, self.x.shape[0]))
        for i in range(self.k):
            distance[i, :] = np.sum((self.x - self.mu[i, :])**2, axis=1)
        
        self.t = np.argmin(distance, axis=0)
        
    def train(self, x):
        self.x = x
        self.mu = np.zeros((self.k, x.shape[1]))
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
            
        return self.t