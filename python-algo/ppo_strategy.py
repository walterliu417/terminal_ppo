import random
from sys import maxsize
import json
import numpy as np
import os
import torch
from torch.distributions import Categorical
import time

from nn_creator import *
import gamelib
from gamelib.util import *



"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""

struct_to_int = {"FF": 1, "EF": 2, "DF": 3}

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self, gamenum):
        super().__init__()
        self.gamenum = gamenum
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

        try:
            open("mefirst.txt", "w").close()
        except Exception as e:
            print(e)
        try:
            open("buffer/temp_data.py", "w").close()
        except Exception as e:
            print(e)
        try:
            open(f"buffer/{self.gamenum}.py", "w").close()
        except Exception as e:
            print(e)
        try:
            open(f"buffer/{self.gamenum}_rewards.txt", "w").close()
        except Exception as e:
            print(e)

        self.building_resource_penalty = 0
        self.unit_resource_penalty = 0
        self.building_reward = 0
        self.unit_reward = 0
        self.scored_locations = np.zeros((2, 28))
            
    def on_game_start(self, config):
        
        """ 
        Read in config and perform any initial setup here 
        """
        try:
            with open("mefirst.txt", "r") as file:
                a = file.read()
            if a == "me first!":
                self.save = False
            else:
                with open("mefirst.txt", "w") as file:
                    file.write("me first!")
                self.save = True
        except:
            with open("mefirst.txt", "w") as file:
                file.write("me first!")
            self.save = True
        
            
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0

        
        self.unit_model = UnitAgent()
        self.building_model = BuildingAgent()
        try:
            self.unit_model.load_state_dict(torch.load("checkpoints/unit_latest.pth", weights_only=True))
            self.building_model.load_state_dict(torch.load("checkpoints/building_latest.pth", weights_only=True))
        except:
            # New model
            pass
        self.unit_model.eval()
        self.building_model.eval()

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        open("inturn.txt", "w").close()

        game_state = gamelib.GameState(self.config, turn_state)

        list_board, list_my_board, list_my_units, list_their_units, list_my_stats, list_their_stats = self.get_current_state_lists(game_state)

        st = f"{list_board},{list_my_board},{list_my_units},{list_their_units},{list_my_stats},{list_their_stats}\n"
        if self.save:
            with open("buffer/temp_data.py","a") as file:
                file.write(st)
                

        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        self.ppo_strategy(game_state)
        #self.starter_strategy(game_state)
        
        game_state.submit_turn()

    """
    NOTE: All the methods after this point are part of the sample starter-algo
    strategy and can safely be replaced for your custom algo.
    """

    def get_current_state_lists(self, game_state):
        board = game_state.game_map.getmap()
        
        list_board = np.zeros((6, 29, 15))
        list_my_board = np.zeros((210, 6))
        list_my_units = np.zeros((3, 28))
        list_their_units = np.zeros((3, 28))

        for x in range(len(board)):
            for y in range(len(board[0])):

                item = board[x][y]

                if item != []:
                    along_coord, across_coord_up = xy_to_diag_coord_full(x, y)
                    item = item[0]
                    if item.player_index == 0:
                        multiplier = 1
                        my_coord = trig_to_tensor[(x, y)]
                    else:
                        multiplier = -1
                        
                    if item.unit_type == "FF":
                        if item.upgraded:
                            list_board[1][along_coord][across_coord_up] = (item.health / 120) * multiplier
                            if multiplier == 1:
                                list_my_board[my_coord][1] = item.health / 120
                        else:
                            list_board[0][along_coord][across_coord_up] = (item.health / 60) * multiplier
                            if multiplier == 1:
                                list_my_board[my_coord][0] = item.health / 60

                    elif item.unit_type == "EF":
                        if item.upgraded:
                            list_board[3][along_coord][across_coord_up] = (item.health / 30) * multiplier
                            if multiplier == 1:
                                list_my_board[my_coord][3] = item.health / 30
                        else:
                            list_board[2][along_coord][across_coord_up] = (item.health / 30) * multiplier
                            if multiplier == 1:
                                list_my_board[my_coord][2] = item.health / 30
                    elif item.unit_type == "DF":
                        if item.upgraded:
                            list_board[5][along_coord][across_coord_up] = (item.health / 75) * multiplier
                            if multiplier == 1:
                                list_my_board[my_coord][5] = item.health / 75
                        else:
                            list_board[4][along_coord][across_coord_up] = (item.health / 75) * multiplier
                            if multiplier == 1:
                                list_my_board[my_coord][4] = item.health / 75
                            

                    elif item.unit_type == "PI":
                        if multiplier == 1:
                            list_my_units[0][x] += 1
                        elif multiplier == -1:
                            list_their_units[0][x] += 1
                    elif item.unit_type == "EI":
                        if multiplier == 1:
                            list_my_units[1][x] += 1
                        elif multiplier == -1:
                            list_their_units[1][x] += 1
                    elif item.unit_type == "SI":
                        if multiplier == 1:
                            list_my_units[2][x] += 1
                        elif multiplier == -1:
                            list_their_units[2][x] += 1

        list_my_stats = [game_state.my_health, game_state.get_resources(0)[0], game_state.get_resources(0)[1]]
        list_their_stats = [game_state.enemy_health, game_state.get_resources(1)[0], game_state.get_resources(1)[1]]
        

        return list_board.tolist(), list_my_board.tolist(), list_my_units.tolist(), list_their_units.tolist(), list_my_stats, list_their_stats
    
    def ppo_strategy(self, game_state):

        friendly_edges = game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
    
        with open("buffer/temp_data.py", "r") as file:
            data = [_.strip() for _ in file.readlines()]
        if len(data) <= 1:
            building_bias_factor, unit_bias_factor = 0, 0
        else:
            self.calc_rewards(data[-2], data[-1])
            self.building_reward -= self.building_resource_penalty
            self.unit_reward -= self.unit_resource_penalty

            with open(f"buffer/{self.gamenum}_rewards.txt", "a") as file:
                file.write(f"{self.building_reward},{self.unit_reward}\n")
            
            
            self.building_reward = 0
            self.unit_reward = 0
            self.scored_locations = np.zeros((2, 28))

            # Some bias?
            building_bias_factor = self.building_resource_penalty / 500
            unit_bias_factor = self.unit_resource_penalty / 500
            
            # No bias.
            #building_bias_factor, unit_bias_factor = 0, 0
                
            self.building_resource_penalty = 0
            self.unit_resource_penalty = 0

        if len(data) <= 1:
            state = eval(data[0])
            current_board = torch.tensor(state[0], dtype=torch.float)
            board = torch.cat((current_board, torch.zeros((6, 29, 15), dtype=torch.float)), dim=0)
            current_my_board = torch.tensor(state[1], dtype=torch.float)
            my_board = torch.cat((current_my_board, torch.zeros((210, 6), dtype=torch.float)), dim=1)
            my_stats = torch.tensor(state[4], dtype=torch.float)
            my_stats_change = torch.tensor(np.zeros(3), dtype=torch.float)
            their_stats = torch.tensor(state[5], dtype=torch.float)
            their_stats_change = torch.tensor(np.zeros(3), dtype=torch.float)
            my_units = torch.zeros((3, 28))
            their_units = torch.zeros((3, 28))
        else:
            last_board, present_board = eval(data[-2]), eval(data[-1])
            current_board = torch.tensor(present_board[0], dtype=torch.float)
            board = torch.cat((torch.tensor(present_board[0], dtype=torch.float), torch.tensor(np.array(present_board[0]) - np.array(last_board[0]), dtype=torch.float)), dim=0)
            my_board = torch.cat((torch.tensor(present_board[1], dtype=torch.float), torch.tensor(np.array(present_board[1]) - np.array(last_board[1]), dtype=torch.float)), dim=1)
            my_stats = torch.tensor(present_board[4], dtype=torch.float)
            my_stats_change = torch.tensor(np.array(present_board[4]) - np.array(last_board[4]), dtype=torch.float)
            their_stats = torch.tensor(present_board[5], dtype=torch.float)
            their_stats_change = torch.tensor(np.array(present_board[5]) - np.array(last_board[5]), dtype=torch.float)
            my_units = torch.tensor(last_board[2], dtype=torch.float)
            their_units = torch.tensor(last_board[3], dtype=torch.float)

        unit_input = torch.cat((my_units, their_units, torch.tensor(self.scored_locations, dtype=torch.float)), dim=0).reshape((1, 8, 28))
        stats = torch.cat((my_stats, my_stats_change, their_stats, their_stats_change)).reshape((1, 12))
        all_building_input = board.reshape((1, 12, 29, 15))
        my_building_input = my_board.reshape((1, 210, 12))

        # Forward pass through the models (policy + value function)
        unit_probs, unit_value = self.unit_model.forward(unit_input, all_building_input, stats)
        building_probs, building_value = self.building_model.forward(unit_input, my_building_input, stats)
        episode_obs = [unit_input.tolist(), all_building_input.tolist(), my_building_input.tolist(), stats.tolist()]
        
        # Sample action from the probability distribution
        # unit_probs -> (1, 28, 3, 10)
        # building_probs -> (1, 210, 6, 2)

        # Attempt to bias the 0 action if there are resource penalties.
        building_bias = torch.zeros_like(building_probs)
        building_bias[:, :, :, 0] += building_bias_factor
        building_probs += building_bias

        unit_bias = torch.zeros_like(unit_probs)
        unit_bias[:, :, :, 0] += unit_bias_factor
        unit_probs += unit_bias

        building_dist = Categorical(probs=building_probs)
        building_action = building_dist.sample()[0] # (210, 6)

        unit_dist = Categorical(probs=unit_probs)
        unit_action = unit_dist.sample()[0] # (28, 3)

        episode_actions = [building_action.tolist(), unit_action.tolist()]
        
        current_mobile_pts = game_state.get_resources(0)[0]
        current_struct_pts = game_state.get_resources(0)[1]
        attempt_mobile_pts, attempt_struct_pts = 0, 0

        # Attempt to mobilise new units
        # Try interceptors first?
        for i in range(len(unit_action)):
            if (unit_action[i][0] != 0):
                for num in range(int(unit_action[i][0])):
                    game_state.attempt_spawn(INTERCEPTOR, friendly_edges[i])
                    attempt_mobile_pts += 1
        # Demolishers.
        for i in range(len(unit_action)):
            if (unit_action[i][1] != 0):
                for num in range(int(unit_action[i][1])):
                    game_state.attempt_spawn(DEMOLISHER, friendly_edges[i])
                    attempt_mobile_pts += 3
        # Scouts
        for i in range(len(unit_action)):
            if (unit_action[i][2] != 0):
                for num in range(int(unit_action[i][2])):
                    game_state.attempt_spawn(SCOUT, friendly_edges[i])
                    attempt_mobile_pts += 1

        # Attempt to build/destroy structures
        # Turrets
        turrets = building_action[:, 4]
        for loc in range(len(turrets)):
            x, y = tensor_to_trig[loc]
            if (turrets[loc] == 1) and (current_board[4][x][y] == 0) and (current_board[5][x][y] == 0):
                game_state.attempt_spawn(TURRET, [x, y])
                attempt_struct_pts += 2
            elif (turrets[loc] == 0) and (current_board[4][x][y] == 1):
                game_state.attempt_remove([x, y])

        upgraded_turrets = building_action[:, 5]
        for loc in range(len(upgraded_turrets)):
            x, y = tensor_to_trig[loc]
            if (upgraded_turrets[loc] == 1) and (current_board[4][x][y] == 0) and (current_board[5][x][y] == 0):
                game_state.attempt_spawn(TURRET, [x, y])
                game_state.attempt_upgrade([x, y])
                attempt_struct_pts += 6
            elif (upgraded_turrets[loc] == 1) and (current_board[4][x][y] != 0) and (current_board[5][x][y] == 0):
                game_state.attempt_upgrade([x, y])
                attempt_struct_pts += 4
            elif (upgraded_turrets[loc] == 0) and (current_board[5][x][y] == 1):
                game_state.attempt_remove([x, y])

        # Walls
        walls = building_action[:, 0]
        for loc in range(len(walls)):
            x, y = tensor_to_trig[loc]
            if (walls[loc] == 1) and (current_board[0][x][y] == 0) and (current_board[1][x][y] == 0):
                game_state.attempt_spawn(WALL, [x, y])
                attempt_struct_pts += 1
            elif (walls[loc] == 0) and (current_board[0][x][y] == 1):
                game_state.attempt_remove([x, y])
            
        upgraded_walls = building_action[:, 1]
        for loc in range(len(upgraded_walls)):
            x, y = tensor_to_trig[loc]
            if (upgraded_walls[loc] == 1) and (current_board[0][x][y] == 0) and (current_board[1][x][y] == 0):
                game_state.attempt_spawn(WALL, [x, y])
                game_state.attempt_upgrade([x, y])
                attempt_struct_pts += 2
            elif (upgraded_walls[loc] == 1) and (current_board[0][x][y] != 0) and (current_board[1][x][y] == 0):
                game_state.attempt_upgrade([x, y])
                attempt_struct_pts += 1
            elif (upgraded_walls[loc] == 0) and (current_board[1][x][y] == 1):
                game_state.attempt_remove([x, y])

        # Supports
        supports = building_action[:, 2]
        for loc in range(len(supports)):
            x, y = tensor_to_trig[loc]
            if (supports[loc] == 1) and (current_board[2][x][y] == 0) and (current_board[3][x][y] == 0):
                game_state.attempt_spawn(SUPPORT, [x, y])
                attempt_struct_pts += 4
            elif (upgraded_walls[loc] == 0) and (current_board[2][x][y] == 1):
                game_state.attempt_remove([x, y])

        upgraded_supports = building_action[:, 3]
        for loc in range(len(upgraded_supports)):
            x, y = tensor_to_trig[loc]
            if (upgraded_supports[loc] == 1) and (current_board[2][x][y] == 0) and (current_board[3][x][y] == 0):
                game_state.attempt_spawn(SUPPORT, [x, y])
                game_state.attempt_upgrade([x, y])
                attempt_struct_pts += 8
            elif (upgraded_supports[loc] == 1) and (current_board[2][x][y] != 0) and (current_board[3][x][y] == 0):
                game_state.attempt_upgrade([x, y])
                attempt_struct_pts += 4
            elif (upgraded_supports[loc] == 0) and (current_board[3][x][y] == 1):
                game_state.attempt_remove([x, y])
        

        self.building_resource_penalty = max(0, attempt_struct_pts - current_struct_pts) / 20
        self.unit_resource_penalty = max(0, attempt_mobile_pts - current_mobile_pts) / 20
                
        # Save data for training.
        episode_building_log_probs = building_dist.log_prob(building_action).tolist()
        episode_unit_log_probs = unit_dist.log_prob(unit_action).tolist()
        episode_log_probs = [episode_building_log_probs, episode_unit_log_probs]
        episode_values = [float(building_value.detach()), float(unit_value.detach())]
        with open(f"buffer/{self.gamenum}.py", "a") as file:
            file.write(f"[{episode_obs},{episode_actions},{episode_log_probs},{episode_values}]\n")
            
    def calc_rewards(self, lap, lbp):
        last_action_phase, last_build_phase = lap, lbp
        _,_, _, _, last_my_stats, last_their_stats = eval(last_action_phase)
        _,_, _, _, my_stats, their_stats = eval(last_build_phase)

        # Calculate advantage from scoring/being scored
        my_health_lost = last_my_stats[0] - my_stats[0]
        their_health_lost = last_their_stats[0] - their_stats[0]
        self.unit_reward += M * np.log(1.0001 - their_health_lost / last_their_stats[0])
        self.building_reward -= M * np.log(1.0001 - my_health_lost / last_my_stats[0])

        
    def on_action_frame(self, turn_string):
        savenow = False
        try:
            with open("inturn.txt", "r") as file:
                a = file.read()
            if a == "first turn done":
                pass
            else:
                savenow = True

        except:
            savenow = True

        if savenow and self.save:
            with open("inturn.txt", "w") as file:
                file.write("first turn done")
            
            game_state = gamelib.GameState(self.config, turn_string)
            list_board, list_my_board, list_my_units, list_their_units, list_my_stats, list_their_stats = self.get_current_state_lists(game_state)
            
            st = f"{list_board},{list_my_board},{list_my_units},{list_their_units},{list_my_stats},{list_their_stats}\n"
            with open("buffer/temp_data.py","a") as file:
                file.write(st)
                
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)
        events = state["events"]
        #with open("fun.txt", "a") as file:
        #    file.write(str(state["events"]) + "\n")
        
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly, 
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                self.scored_locations[0][location[0]] += 1/30
            else:
                self.scored_locations[1][location[0]] += 1/30
            
        damages = events["damage"]
        for dmg in damages:
            multiplier = 1 if dmg[4] == 1 else -1
            thing_type = dmg[2]
            dmg_val = dmg[1]
            if thing_type == 0:
                # wall
                self.unit_reward += (dmg_val / 60) * WALL_POINT * multiplier
            elif thing_type == 1:
                # support
                self.unit_reward += (dmg_val / 30) * SUPPORT_POINT * multiplier
            elif thing_type == 2:
                # turret
                self.unit_reward += (dmg_val / 75) * TURRET_POINT * multiplier
            elif thing_type == 3:
                # scout
                if dmg_val == 20:
                    # damaged by interceptor
                    self.unit_reward += (dmg_val / 15) * MOBILE_POINTS * multiplier
                else:
                    self.building_reward += (dmg_val / 15) * MOBILE_POINTS * multiplier
            elif thing_type == 4:
                # destructor
                if dmg_val == 20:
                    self.unit_reward += (dmg_val / 5) * MOBILE_POINTS * multiplier
                else:
                    self.building_reward += (dmg_val / 5) * MOBILE_POINTS * multiplier
            elif thing_type == 5:
                # interceptor
                if dmg_val == 20:
                    self.unit_reward += (dmg_val / 40) * MOBILE_POINTS * multiplier
                else:
                    self.building_reward += (dmg_val / 40) * MOBILE_POINTS * multiplier

        shields = events["shield"]
        for shield in shields:
            if shield[4] == 1:
                self.building_reward += (shield[1] / 20) * MOBILE_POINTS


            


if __name__ == "__main__":
    with open("thegame.txt", "r") as file:
        num = int(file.read().strip())
    algo = AlgoStrategy(num)
    algo.start()
