import sys
import math



BANNER_TEXT = "---------------- Starting Your Algo --------------------"

# Parameters to tune
ADV_PERCENT_POINT = 3
M = ADV_PERCENT_POINT / math.log(1.0001 - 1/30)
MOBILE_POINTS = 2
WALL_POINT = 0.5
TURRET_POINT = 6
SUPPORT_POINT = 2

VICTORY_REWARD = 250



def get_command():
    """Gets input from stdin

    """
    try:
        ret = sys.stdin.readline()
    except EOFError:
        # Game parent process terminated so exit
        debug_write("Got EOF, parent game process must have died, exiting for cleanup")
        exit()
    if ret == "":
        # Happens if parent game process dies, so exit for cleanup, 
        # Don't change or starter-algo process won't exit even though the game has closed
        debug_write("Got EOF, parent game process must have died, exiting for cleanup")
        exit()
    return ret

def send_command(cmd):
    """Sends your turn to standard output.
    Should usually only be called by 'GameState.submit_turn()'

    """
    sys.stdout.write(cmd.strip() + "\n")
    sys.stdout.flush()

def debug_write(*msg):
    """Prints a message to the games debug output

    Args:
        msg: The message to output

    """
    #Printing to STDERR is okay and printed out by the game but doesn't effect turns.
    sys.stderr.write(", ".join(map(str, msg)).strip() + "\n")
    sys.stderr.flush()
