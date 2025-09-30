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

# Build the mapping
tensor_to_trig = dict()

pts = 0
for x in range(28):
    if (0 <= x <= 13):
        for y in range(13 - x, 14):
            tensor_to_trig[pts] = (x, y)
            pts += 1
    elif (14 <= x <= 27):
        for y in range(x - 14, 14):
            tensor_to_trig[pts] = (x, y)
            pts += 1

trig_to_tensor = dict()

for key, value in list(tensor_to_trig.items()):
    trig_to_tensor[value] = key

def xy_to_diag_coord_full(x, y):
    along_up_coord = (y + x - 13)
    # across_line : y = -x + (x_p+y_p)
    # along_up_line_odd : y = x + 13
    # along_up_line_even : y = x + 14
    if (x + y) % 2 == 0:
        # even
        x_border = (x + y - 14) / 2
    else:
        # odd
        x_border = (x + y - 13) / 2
    across_coord = x - x_border
    # Converted to size (29, 15)
    return int(along_up_coord), int(across_coord)

def diag_coord_to_xy_full(along_up_coord, across_coord):
    if (along_up_coord - 2 * across_coord) % 2 == 0:
        y = (1/2) * (along_up_coord - 2 * across_coord + 26)
        x = along_up_coord + 13 - y
    else:
        y = (1/2) * (along_up_coord - 2 * across_coord  + 27)
        x = along_up_coord + 13 - y
    return int(x), int(y)



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
