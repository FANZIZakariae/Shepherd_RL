import numpy as np

class RuleBasedShepherd:
    def __init__(
        self,
        guide_offset=0.10,
        arrive_threshold=0.05
    ):
        """
        guide_offset: how far BEHIND the sheep (relative to goal direction)
        arrive_threshold: distance at which sheep is considered 'done'
        """
        self.guide_offset = guide_offset
        self.arrive_threshold = arrive_threshold

    def act(self, obs):
        """
        obs: flattened observation vector
        returns: action angle in degrees [-180, +180]
        """

        obs = np.array(obs)
        n_sheep = (len(obs) - 4) // 4 # 4 values per sheep (-4 for goal and obstacle)

        sheep_positions = []
        goal_positions = []

        shepherd_pos = np.zeros(2)  # shepherd is origin in relative coords

        # --- Reconstruct absolute positions ---
        for i in range(n_sheep):
            sheep_rel = obs[i * 4 : i * 4 + 2]
            goal_rel  = obs[i * 4 + 2 : i * 4 + 4]

            sheep_pos = shepherd_pos + sheep_rel
            goal_pos  = sheep_pos + goal_rel

            sheep_positions.append(sheep_pos)
            goal_positions.append(goal_pos)

        sheep_positions = np.array(sheep_positions)
        goal_positions = np.array(goal_positions)

        # --- Compute distances to goal ---
        dists = np.linalg.norm(sheep_positions - goal_positions, axis=1)

        active = dists > self.arrive_threshold
        if not np.any(active):
            return np.array([0.0], dtype=np.float32)  # arbitrary heading

        # --- Select furthest sheep ---
        idx = np.argmax(dists * active)
        sheep = sheep_positions[idx]
        goal = goal_positions[idx]

        # --- Compute guiding direction ---
        goal_dir = goal - sheep
        norm = np.linalg.norm(goal_dir)
        if norm < 1e-6:
            return np.array([0.0], dtype=np.float32)

        goal_dir /= norm

        # --- Desired shepherd position (behind sheep) ---
        guide_pos = sheep - goal_dir * self.guide_offset

        # --- Desired movement direction ---
        move_vec = guide_pos - shepherd_pos
        move_norm = np.linalg.norm(move_vec)

        if move_norm < 1e-6:
            return np.array([0.0], dtype=np.float32)

        move_vec /= move_norm

        # --- Convert direction vector to angle ---
        angle_rad = np.arctan2(move_vec[1], move_vec[0])
        angle_deg = np.rad2deg(angle_rad)/180  # Scale to [-1, +1]

        # Normalize to [-180, +180]
        angle_deg = ((angle_deg + 1) % 2) - 1

        return np.array([angle_deg], dtype=np.float32)


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
    

class TipsyShepherd:
    def __init__(self):
        pass

    def act(self, obs):
        """
        Returns a random orientation angle in degrees (*180) [-1, +1]
        """
        angle = np.random.uniform(-1, 1)
        return np.array([angle], dtype=np.float32)
    

