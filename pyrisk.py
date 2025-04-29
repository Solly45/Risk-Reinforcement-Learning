from game_statistics import GameStatistics
import logging
import random
import importlib
import re
import collections
import curses
import csv
import os
from game import Game

from world import CONNECT, MAP, KEY, AREAS

LOG = logging.getLogger("pyrisk")
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--nocurses", dest="curses", action="store_false", default=True, help="Disable the ncurses map display")
parser.add_argument("--nocolor", dest="color", action="store_false", default=True, help="Display the map without colors")
parser.add_argument("-l", "--log", action="store_true", default=False, help="Write game events to a logfile")
parser.add_argument("-d", "--delay", type=float, default=0, help="Delay in seconds after each action is displayed")
parser.add_argument("-s", "--seed", type=int, default=None, help="Random number generator seed")
parser.add_argument("-g", "--games", type=int, default=1, help="Number of rounds to play")
parser.add_argument("-w", "--wait", action="store_true", default=False, help="Pause and wait for a keypress after each action")
parser.add_argument("players", nargs="+", help="Names of the AI classes to use. May use 'ExampleAI*3' syntax.")
parser.add_argument("--deal", action="store_true", default=False, help="Deal territories rather than letting players choose")

args = parser.parse_args()

NAMES = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT"]
csv_filename = "data/alpha_wins.csv"
turn_csv_filename = "data/turn_stats.csv"
file_exists = os.path.isfile(csv_filename)
turn_file_exists = os.path.isfile(turn_csv_filename)

LOG.setLevel(logging.DEBUG)
if args.log:
    logging.basicConfig(filename="pyrisk.log", filemode="w")
elif not args.curses:
    logging.basicConfig()

if args.seed is not None:
    random.seed(args.seed)

player_classes = []
for p in args.players:
    match = re.match(r"(\w+)?(\*\d+)?", p)
    if match:
        name = match.group(1)
        package = name[:-2].lower()
        count = int(match.group(2)[1:]) if match.group(2) else 1
        try:
            klass = getattr(importlib.import_module("ai."+package), name)
            for i in range(count):
                player_classes.append(klass)
        except:
            print("Unable to import AI %s from ai/%s.py" % (name, package))
            raise

kwargs = dict(curses=args.curses, color=args.color, delay=args.delay,
              connect=CONNECT, cmap=MAP, ckey=KEY, areas=AREAS, wait=args.wait, deal=args.deal)

def wrapper(stdscr, **kwargs):
    g = Game(screen=stdscr, **kwargs)
    for i, klass in enumerate(player_classes):
        g.add_player(NAMES[i], klass)
    return g.play(), g.stats.turn_counts[-1]  # Return both winner and turn count

if args.games == 1:
    if args.curses:
        wrapper_output = curses.wrapper(wrapper, **kwargs)
    else:
        wrapper_output = wrapper(None, **kwargs)
    victor, turns = wrapper_output
else:
    wins = collections.defaultdict(int)
    alpha_wins_last_10 = 0
    turn_counts = []

# Track wins over 100 games
alpha_wins_last_100 = 0
hundred_games_csv = "data/100gameswins.csv"
hundred_games_file_exists = os.path.isfile(hundred_games_csv)

for j in range(args.games):
    
    kwargs['round'] = (j+1, args.games)
    kwargs['history'] = wins

    if args.curses:
        wrapper_output = curses.wrapper(wrapper, **kwargs)
    else:
        wrapper_output = wrapper(None, **kwargs)
    
    victor, turns = wrapper_output  # Get winner and turn count
    wins[victor] += 1
    
    turn_counts.append(turns)  # Store turn count

    if victor == "ALPHA":
        alpha_wins_last_10 += 1  # Track ALPHA's wins for last 10 games
        alpha_wins_last_100 += 1  # Track ALPHA's wins for last 100 games

    # Every 10 games, log ALPHA's win count and turn average
    if (j + 1) % 10 == 0:
        # Log ALPHA wins every 10 games
        with open(csv_filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Game Range", "ALPHA Wins"])
                file_exists = True
            writer.writerow([f"Games {j - 9}-{j + 1}", alpha_wins_last_10])

        print(f"Logged ALPHA's wins for games {j - 9}-{j + 1}.")

        # Log Average Turns
        avg_turns = sum(turn_counts[-10:]) / 10  # Compute average turns for last 10 games
        with open(turn_csv_filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not turn_file_exists:
                writer.writerow(["Game Range", "Average Turns"])
                turn_file_exists = True
            writer.writerow([f"Games {j - 9}-{j + 1}", avg_turns])

        print(f"Logged Average Turns for games {j - 9}-{j + 1}: {avg_turns:.2f} turns.")

        alpha_wins_last_10 = 0  # Reset ALPHA win counter

    # Every 100 games, log ALPHA win rate
    if (j + 1) % 100 == 0:
        win_rate_100 = (alpha_wins_last_100 / 100) * 100  # Calculate win rate percentage

        with open(hundred_games_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not hundred_games_file_exists:
                writer.writerow(["Game Range", "Win Rate (%)"])
                hundred_games_file_exists = True
            writer.writerow([f"Games {j - 99}-{j + 1}", f"{win_rate_100:.2f}"])

        print(f"Logged ALPHA's win rate for games {j - 99}-{j + 1}: {win_rate_100:.2f}%.")

        alpha_wins_last_100 = 0  # Reset 100-game win tracker

# Save final game statistics
if hasattr(Game, 'stats'):
    print("Saving game statistics to JSON file...")
    Game.stats.save_to_json("game_stats.json")
    print("Game statistics saved successfully.")
else:
    print("No game statistics available to save.")
