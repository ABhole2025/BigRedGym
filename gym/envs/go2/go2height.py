from gym.envs.go2.go2 import Go2


class Go2Height(Go2):
    def _reward_tracking_base_height(self):
        """Reward tracking the commanded base height."""
        error = (self.base_height - self.height_command).flatten()
        error = error / self.scales["base_height"]
        return self._sqrdexp(error)
