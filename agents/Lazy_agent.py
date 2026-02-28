import numpy as np

class LazyShepherd:
    def __init__(self):
        self.angle = None

    def reset(self):
        """
        Must be called at the beginning of each episode
        """
        self.angle = np.random.uniform(-1, 1)

    def act(self, obs):
        """
        Returns the same orientation angle throughout the episode
        """
        if self.angle is None:
            self.reset()

        return np.array([self.angle], dtype=np.float32)
    
