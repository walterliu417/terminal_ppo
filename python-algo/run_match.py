import os
import glob
import subprocess
import sys

from gamelib.util import *

# Runs a single game
def run_single_game(process_command):
    print("Start run a match")
    p = subprocess.Popen(
        process_command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=sys.stdout
        )
    # daemon necessary so game shuts down if this script is shut down by user
    p.daemon = 1
    p.wait()

def run_match(algo1_path, algo2_path):
    
    files = glob.glob('replays/*')
    for f in files:
        os.remove(f)
    try:
        os.remove("mefirst.txt")
    except:
        pass
    try:
        os.remove("buffer/temp_data.py")
    except:
        pass

    # Get location of this run file
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.join(file_dir, os.pardir)
    parent_dir = os.path.abspath(parent_dir)
    
    # Get if running in windows OS
    is_windows = sys.platform.startswith('win')
    #print("Is windows: {}".format(is_windows))
    
    algo1 = parent_dir + "/" + algo1_path
    algo2 = parent_dir + "/" + algo2_path
    
    # If folder path is given instead of run file path, add the run file to the path based on OS
    # trailing_char deals with if there is a trailing \ or / or not after the directory name
    
    #print("Algo 1: ", algo1)
    #print("Algo 2:", algo2)
    
    run_single_game("cd {} && java -jar engine.jar work {} {}".format(parent_dir, algo1, algo2))


    filename = glob.glob('replays/*')[0]
    with open(filename, "r") as file:
        a = file.readlines()
    endframe = a[-1]
    p1index = endframe.find("p1Stats")
    trnc1 = endframe[p1index + 10:]
    p1indexend = trnc1.find(",")
    p1endhealth = float(trnc1[:p1indexend-1])
    p2index = endframe.find("p2Stats")
    trnc2 = endframe[p2index + 10:]
    p2indexend = trnc2.find(",")
    p2endhealth = float(trnc1[:p2indexend-1])

    with open("thegame.txt", "r") as file:
        num = int(file.read().strip())

    if p1endhealth > p2endhealth:
        with open(f"buffer/{num}_rewards.txt", "a") as file:
            file.write(f"{VICTORY_REWARD}\n")
    else:
        with open(f"buffer/{num}_rewards.txt", "a") as file:
            file.write(f"-{VICTORY_REWARD}\n")



if __name__ == "__main__":
    run_match("python-algo/ppo_strategy.ps1", "python-algo/starter_strategy.ps1")