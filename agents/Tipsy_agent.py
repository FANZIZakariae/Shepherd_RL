import numpy as np

class TipsyShepherd:
    def __init__(self):
        pass

    def act(self, obs):
        """
        Returns a random orientation angle in degrees (*180) [-1, +1]
        """
        angle = np.random.uniform(-1, 1)
        return np.array([angle], dtype=np.float32)
    