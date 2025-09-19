import random
from sys import maxsize
import json
import numpy as np
import os
import torch
from torch.distributions import Categorical

from nn_creator import TerminalA2C
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

        self.model = TerminalA2C()
        try:
            self.model.load_state_dict(torch.load("checkpoints/latest.pth"))
        except:
            # New model
            pass

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
        # This is a good place to do initial setup
        self.scored_on_locations = []

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

        list_board, list_my_stats, list_their_stats = self.get_current_state_lists(game_state)
        

        st = f"{list_board},{list_my_stats},{list_their_stats}\n"
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
        
        list_board = np.zeros((6, 28, 28))

        for x in range(len(board)):
            for y in range(len(board[0])):
                item = board[x][y]
                if item != []:
                    item = item[0]
                    if item.player_index == 0:
                        multiplier = 1
                    else:
                        multiplier = -1
                        
                    if item.unit_type == "FF":
                        if item.upgraded:
                            list_board[1][x][y] = 2 * (item.health / 120) * multiplier
                        else:
                            list_board[0][x][y] = (item.health / 60) * multiplier

                    elif item.unit_type == "EF":
                        if item.upgraded:
                            list_board[3][x][y] = (item.health / 30) * multiplier
                        else:
                            list_board[2][x][y] = (item.health / 30) * multiplier
                    elif item.unit_type == "DF":
                        if item.upgraded:
                            list_board[5][x][y] = (item.health / 75) * multiplier
                        else:
                            list_board[4][x][y] = (item.health / 75) * multiplier
                        
                        

        list_my_stats = [game_state.my_health, game_state.get_resources(0)[0], game_state.get_resources(0)[1]]
        list_their_stats = [game_state.enemy_health, game_state.get_resources(1)[0], game_state.get_resources(1)[1]]
        
        return list_board.tolist(), list_my_stats, list_their_stats
    
    def ppo_strategy(self, game_state):

        friendly_edges = game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
    
        with open("buffer/temp_data.py", "r") as file:
            data = [_.strip() for _ in file.readlines()]
        if len(data) <= 1:
            episode_rewards = 0
            building_bias_factor, unit_bias_factor = 0, 0
        else:
            episode_rewards = self.calc_reward(data[-3], data[-2], data[-1]) - (self.building_resource_penalty + self.unit_resource_penalty)

            with open(f"buffer/{self.gamenum}_rewards.txt", "a") as file:
                file.write(str(episode_rewards) + "\n")

            building_bias_factor = self.building_resource_penalty / 250
            unit_bias_factor = self.unit_resource_penalty / 250
            with open("fun.txt", "a") as file:
                file.write(f"{self.building_resource_penalty, building_bias_factor, self.unit_resource_penalty, unit_bias_factor}")
                
            self.building_resource_penalty = 0
            self.unit_resource_penalty = 0

        if len(data) <= 1:
            state = eval(data[0])
            current_board = torch.tensor(state[0], dtype=torch.float)
            board = torch.row_stack((current_board, torch.zeros((6, 28, 28), dtype=torch.float)))
            my_stats = torch.tensor(state[1], dtype=torch.float)
            my_stats_change = torch.tensor(np.zeros(3), dtype=torch.float)
            their_stats = torch.tensor(state[2], dtype=torch.float)
            their_stats_change = torch.tensor(np.zeros(3), dtype=torch.float)
        else:
            last_board, present_board = eval(data[-2]), eval(data[-1])
            current_board = torch.tensor(present_board[0], dtype=torch.float)
            board = torch.row_stack((torch.tensor(present_board[0], dtype=torch.float), torch.tensor(np.array(present_board[0]) - np.array(last_board[0]), dtype=torch.float)))
            my_stats = torch.tensor(present_board[1], dtype=torch.float)
            my_stats_change = torch.tensor(np.array(present_board[1]) - np.array(last_board[1]), dtype=torch.float)
            their_stats = torch.tensor(present_board[2], dtype=torch.float)
            their_stats_change = torch.tensor(np.array(present_board[2]) - np.array(last_board[2]), dtype=torch.float)

        # Forward pass through the model (policy + value function)
        building_probs, unit_probs, value = self.model.forward(my_stats, my_stats_change, their_stats, their_stats_change, board)
        episode_obs = [my_stats.tolist(), my_stats_change.tolist(), their_stats.tolist(), their_stats_change.tolist(), board.tolist()]
        
        # Sample action from the probability distribution

        # Attempt to bias the 0 action if there are resource penalties.
        building_bias = torch.zeros_like(building_probs)
        building_bias[:, :, :, 0] += building_bias_factor
        building_probs += building_bias

        unit_bias = torch.zeros_like(unit_probs)
        unit_bias[:, :, :, 0] += unit_bias_factor
        unit_probs += unit_bias

        building_dist = Categorical(probs=building_probs)
        building_action = building_dist.sample()[0]
        building_action_map = building_action.view((3, 28, 14))

        unit_dist = Categorical(probs=unit_probs)
        unit_action = unit_dist.sample()[0]

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

        # Attempt to build structures
        # Don't destroy upgraded buildings, they are valuable.
        # Turrets
        turrets = building_action_map[2]
        for x in range(len(turrets)):
            for y in range(len(turrets[0])):
                if (turrets[x][y] == 1) and (current_board[4][x][y] == 0) and (current_board[5][x][y] == 0):
                    game_state.attempt_spawn(TURRET, [x, y])
                    attempt_struct_pts += 2
                elif (turrets[x][y] == 0) and (current_board[4][x][y] != 0):
                    game_state.attempt_remove([x, y])
                    attempt_struct_pts += 0.5
                elif (turrets[x][y] == 2) and (current_board[4][x][y] == 0) and (current_board[5][x][y] == 0):
                    game_state.attempt_spawn(TURRET, [x, y])
                    game_state.attempt_upgrade([x, y])
                    attempt_struct_pts += 6
                elif (turrets[x][y] == 2) and (current_board[4][x][y] != 0) and (current_board[5][x][y] == 0):
                    game_state.attempt_upgrade([x, y])
                    attempt_struct_pts += 4
        # Walls
        walls = building_action_map[0]
        for x in range(len(walls)):
            for y in range(len(walls[0])):
                if (walls[x][y] == 1) and (current_board[0][x][y] == 0) and (current_board[1][x][y] == 0):
                    game_state.attempt_spawn(WALL, [x, y])
                    attempt_struct_pts += 1
                elif (walls[x][y] == 0) and (current_board[0][x][y] != 0):
                    game_state.attempt_remove([x, y])
                    attempt_struct_pts += 0.25
                elif (walls[x][y] == 2) and (current_board[0][x][y] == 0) and (current_board[1][x][y] == 0):
                    game_state.attempt_spawn(WALL, [x, y])
                    game_state.attempt_upgrade([x, y])
                    attempt_struct_pts += 2
                elif (walls[x][y] == 2) and (current_board[0][x][y] != 0) and (current_board[1][x][y] == 0):
                    game_state.attempt_upgrade([x, y])
                    attempt_struct_pts += 1

        # Supports
        supports = building_action_map[1]
        for x in range(len(supports)):
            for y in range(len(supports[0])):
                if (supports[x][y] == 1) and (current_board[2][x][y] == 0) and (current_board[3][x][y] == 0):
                    game_state.attempt_spawn(SUPPORT, [x, y])
                    attempt_struct_pts += 4
                elif (supports[x][y] == 0) and (current_board[2][x][y] != 0):
                    game_state.attempt_remove([x, y])
                    attempt_struct_pts += 1
                elif (supports[x][y] == 2) and (current_board[2][x][y] == 0) and (current_board[3][x][y] == 0):
                    game_state.attempt_spawn(SUPPORT, [x, y])
                    game_state.attempt_upgrade([x, y])
                    attempt_struct_pts += 8
                elif (supports[x][y] == 2) and (current_board[2][x][y] != 0) and (current_board[3][x][y] == 0):
                    game_state.attempt_upgrade([x, y])
                    attempt_struct_pts += 4
        

        self.building_resource_penalty = max(0, attempt_struct_pts - current_struct_pts) / 20
        self.unit_resource_penalty = max(0, attempt_mobile_pts - current_mobile_pts) / 20
                
        # Save data for training.
        episode_building_log_probs = building_dist.log_prob(building_action).tolist()
        episode_unit_log_probs = unit_dist.log_prob(unit_action).tolist()
        episode_log_probs = [episode_building_log_probs, episode_unit_log_probs]
        episode_values = float(value)
        with open(f"buffer/{self.gamenum}.py", "a") as file:
            file.write(f"[{episode_obs},{episode_actions},{episode_log_probs},{episode_values}]\n")


    def starter_strategy(self, game_state):
        """
        For defense we will use a spread out layout and some interceptors early on.
        We will place turrets near locations the opponent managed to score on.
        For offense we will use long range demolishers if they place stationary units near the enemy's front.
        If there are no stationary units to attack in the front, we will send Scouts to try and score quickly.
        """
        # First, place basic defenses
        self.build_defences(game_state)
        # Now build reactive defenses based on where the enemy scored
        self.build_reactive_defense(game_state)

        # If the turn is less than 5, stall with interceptors and wait to see enemy's base
        if game_state.turn_number < 5:
            self.stall_with_interceptors(game_state)
        else:
            # Now let's analyze the enemy base to see where their defenses are concentrated.
            # If they have many units in the front we can build a line for our demolishers to attack them at long range.
            if self.detect_enemy_unit(game_state, unit_type=None, valid_x=None, valid_y=[14, 15]) > 10:
                self.demolisher_line_strategy(game_state)
            else:
                # They don't have many units in the front so lets figure out their least defended area and send Scouts there.

                # Only spawn Scouts every other turn
                # Sending more at once is better since attacks can only hit a single scout at a time
                if game_state.turn_number % 2 == 1:
                    # To simplify we will just check sending them from back left and right
                    scout_spawn_location_options = [[13, 0], [14, 0]]
                    best_location = self.least_damage_spawn_location(game_state, scout_spawn_location_options)
                    game_state.attempt_spawn(SCOUT, best_location, 1000)

                # Lastly, if we have spare SP, let's build some supports
                support_locations = [[13, 2], [14, 2], [13, 3], [14, 3]]
                game_state.attempt_spawn(SUPPORT, support_locations)

    def calc_reward(self, flbp, lap, lbp):
        further_last_build_phase, last_action_phase, last_build_phase = flbp, lap, lbp
        _, further_last_my_stats, further_last_their_stats = eval(further_last_build_phase)
        last_buildings, last_my_stats, last_their_stats = eval(last_action_phase)
        buildings, my_stats, their_stats = eval(last_build_phase)
        advantage = 0

        # Calculate advantage from scoring/being scored
        my_health_lost = last_my_stats[0] - my_stats[0]
        their_health_lost = last_their_stats[0] - their_stats[0]
        advantage += M * np.log(1.0001 - their_health_lost / last_their_stats[0])
        advantage -= M * np.log(1.0001 - my_health_lost / last_my_stats[0])

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

    def build_defences(self, game_state):
        """
        Build basic defenses using hardcoded locations.
        Remember to defend corners and avoid placing units in the front where enemy demolishers can attack them.
        """
        # Useful tool for setting up your base locations: https://www.kevinbai.design/terminal-map-maker
        # More community tools available at: https://terminal.c1games.com/rules#Download

        # Place turrets that attack enemy units
        turret_locations = [[0, 13], [27, 13], [8, 11], [19, 11], [13, 11], [14, 11]]
        # attempt_spawn will try to spawn units if we have resources, and will check if a blocking unit is already there
        game_state.attempt_spawn(TURRET, turret_locations)
        
        # Place walls in front of turrets to soak up damage for them
        wall_locations = [[8, 12], [19, 12]]
        game_state.attempt_spawn(WALL, wall_locations)
        # upgrade walls so they soak more damage
        game_state.attempt_upgrade(wall_locations)

    def build_reactive_defense(self, game_state):
        """
        This function builds reactive defenses based on where the enemy scored on us from.
        We can track where the opponent scored by looking at events in action frames 
        as shown in the on_action_frame function
        """
        for location in self.scored_on_locations:
            # Build turret one space above so that it doesn't block our own edge spawn locations
            build_location = [location[0], location[1]+1]
            game_state.attempt_spawn(TURRET, build_location)

    def stall_with_interceptors(self, game_state):
        """
        Send out interceptors at random locations to defend our base from enemy moving units.
        """
        # We can spawn moving units on our edges so a list of all our edge locations
        friendly_edges = game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
        
        # Remove locations that are blocked by our own structures 
        # since we can't deploy units there.
        deploy_locations = self.filter_blocked_locations(friendly_edges, game_state)
        
        # While we have remaining MP to spend lets send out interceptors randomly.
        while game_state.get_resource(MP) >= game_state.type_cost(INTERCEPTOR)[MP] and len(deploy_locations) > 0:
            # Choose a random deploy location.
            deploy_index = random.randint(0, len(deploy_locations) - 1)
            deploy_location = deploy_locations[deploy_index]
            
            game_state.attempt_spawn(INTERCEPTOR, deploy_location)
            """
            We don't have to remove the location since multiple mobile 
            units can occupy the same space.
            """

    def demolisher_line_strategy(self, game_state):
        """
        Build a line of the cheapest stationary unit so our demolisher can attack from long range.
        """
        # First let's figure out the cheapest unit
        # We could just check the game rules, but this demonstrates how to use the GameUnit class
        stationary_units = [WALL, TURRET, SUPPORT]
        cheapest_unit = WALL
        for unit in stationary_units:
            unit_class = gamelib.GameUnit(unit, game_state.config)
            if unit_class.cost[game_state.MP] < gamelib.GameUnit(cheapest_unit, game_state.config).cost[game_state.MP]:
                cheapest_unit = unit

        # Now let's build out a line of stationary units. This will prevent our demolisher from running into the enemy base.
        # Instead they will stay at the perfect distance to attack the front two rows of the enemy base.
        for x in range(27, 5, -1):
            game_state.attempt_spawn(cheapest_unit, [x, 11])

        # Now spawn demolishers next to the line
        # By asking attempt_spawn to spawn 1000 units, it will essentially spawn as many as we have resources for
        game_state.attempt_spawn(DEMOLISHER, [24, 10], 1000)

    def least_damage_spawn_location(self, game_state, location_options):
        """
        This function will help us guess which location is the safest to spawn moving units from.
        It gets the path the unit will take then checks locations on that path to 
        estimate the path's damage risk.
        """
        damages = []
        # Get the damage estimate each path will take
        for location in location_options:
            path = game_state.find_path_to_edge(location)
            damage = 0
            for path_location in path:
                # Get number of enemy turrets that can attack each location and multiply by turret damage
                damage += len(game_state.get_attackers(path_location, 0)) * gamelib.GameUnit(TURRET, game_state.config).damage_i
            damages.append(damage)
        
        # Now just return the location that takes the least damage
        return location_options[damages.index(min(damages))]

    def detect_enemy_unit(self, game_state, unit_type=None, valid_x = None, valid_y = None):
        total_units = 0
        for location in game_state.game_map:
            if game_state.contains_stationary_unit(location):
                for unit in game_state.game_map[location]:
                    if unit.player_index == 1 and (unit_type is None or unit.unit_type == unit_type) and (valid_x is None or location[0] in valid_x) and (valid_y is None or location[1] in valid_y):
                        total_units += 1
        return total_units
        
    def filter_blocked_locations(self, locations, game_state):
        filtered = []
        for location in locations:
            if not game_state.contains_stationary_unit(location):
                filtered.append(location)
        return filtered

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
            list_board, list_my_stats, list_their_stats = self.get_current_state_lists(game_state)
            
            st = f"{list_board},{list_my_stats},{list_their_stats}\n"
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
        
        
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly, 
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))


if __name__ == "__main__":
    with open("thegame.txt", "r") as file:
        num = int(file.read().strip())
    algo = AlgoStrategy(num)
    algo.start()
