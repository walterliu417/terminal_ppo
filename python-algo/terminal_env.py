import numpy as np
import gymnasium as gym

from gamelib.util import *

class MyGameEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "buildings": gym.spaces.Box(-2, 2, shape=([5, 28, 28]), dtype=np.float32),   # [x, y] coordinates
                "map_change": gym.spaces.Box(-2, 2, shape=([5, 28, 28]), dtype=np.float32),   # [x, y] coordinates
                "my_stats": gym.spaces.Box(0, 100, shape=([3]), dtype=np.float32), # Health, Structure Pts, Mobile Pts
                "my_stats_change": gym.spaces.Box(-100, 100, shape=([3]), dtype=np.float32),
                "their_stats": gym.spaces.Box(0, 100, shape=([3]), dtype=np.float32),
                "their_stats_change": gym.spaces.Box(-100, 100, shape=([3]), dtype=np.float32)
            }
        )

        self.action_space = gym.spaces.Dict(
            {
                "new_buildings": gym.spaces.Box(0, 2, shape=([5, 28, 14]), dtype=int),
                "new_units": gym.spaces.Box(0, 30, shape=([3, 28]), dtype=int) # Interceptor SI, Demolisher EI, Scout PI
            }
        )

    def reset(self):
        pass

    def step(self, action):
        next_state, reward, done, info = 0, 0, 0, 0
        return next_state, reward, done, info
    
    def calc_reward(self, last_three_turns):
        further_last_build_phase, last_action_phase, last_build_phase = last_three_turns
        _, further_last_my_stats, further_last_their_stats = eval(further_last_build_phase)
        last_buildings, last_my_stats, last_their_stats = eval(last_action_phase)
        buildings, my_stats, their_stats = eval(last_build_phase)
        advantage = 0

        # Calculate advantage from scoring/being scored
        my_health_lost = last_my_stats[0] - my_stats[0]
        their_health_lost = last_their_stats[0] - their_stats[0]
        advantage += M * np.log(1.0001 - their_health_lost / last_their_stats[0])
        advantage -= M * np.log(1.0001 - my_health_lost / my_stats[0])

        # Calculate advantage from destroying enemy mobile units
        my_mobile_points_used = further_last_my_stats[2] - last_my_stats[2]
        their_mobile_points_used = further_last_their_stats[2] - last_their_stats[2]
        my_lost_mobile_points = my_mobile_points_used - their_health_lost
        their_lost_mobile_points = their_mobile_points_used - my_health_lost
        advantage += (their_lost_mobile_points - my_lost_mobile_points) * MOBILE_POINTS
        
        # Calculate advantage from destroying buildings
        # Walls
        last_walls = np.array(last_buildings[0])
        walls = np.array(buildings[0])
        advantage += np.sum(walls - last_walls) * WALL_POINT
        # Supports
        last_supports = np.array(last_buildings[1])
        supports = np.array(buildings[1])
        advantage += np.sum(supports - last_supports) * SUPPORT_POINT
        last_upgraded_supports = np.array(last_buildings[2])
        upgraded_supports = np.array(buildings[2])
        advantage += np.sum(upgraded_supports - last_upgraded_supports) * SUPPORT_POINT * 2
        # Turrets
        last_turrets = np.array(last_buildings[3])
        turrets = np.array(buildings[3])
        advantage += np.sum(turrets - last_turrets) * TURRET_POINT
        last_upgraded_turrets = np.array(last_buildings[4])
        upgraded_turrets = np.array(buildings[4])
        advantage += np.sum(upgraded_turrets - last_upgraded_turrets) * TURRET_POINT * 2

        return advantage