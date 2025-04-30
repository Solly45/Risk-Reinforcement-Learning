# PyRisk + SkynetAI

A modified version of the open-source PyRisk framework featuring a custom Deep Reinforcement Learning agent, SkynetAI, developed for my final-year dissertation at the University of Stirling (April 2025).

## About the Project

The goal of this project was to build and evaluate a Deep Reinforcement Learning agent (SkynetAI) capable of learning to play a stochastic board game with incomplete information: Risk. The implementation is based on the PyRisk framework, which provides a minimal yet functional Risk game engine in Python.

The new agent uses:
- A Double DQN with Prioritized Experience Replay
- Custom reward shaping for different game phases
- Evaluation against heuristic agents (e.g., AlAI)

The agent and training setup can be found in the `ai/skynet.py` and `train/` folders.


## How to Run

To simulate 5000 games between SkynetAI and AlAI:

```bash
python -m pyrisk SkynetAI AlAI --games=5000
```

To watch a single game play out in terminal mode:

```bash
python -m pyrisk.py SkynetAI AlAI
```

## Attribution

This project is based on the [PyRisk](https://github.com/chronitis/pyrisk) framework created by [Chronitis](https://github.com/chronitis). All original credit for the base implementation of the Risk game in Python goes to them. Significant modifications have been made to implement and evaluate a Deep Q-Learning AI agent.

## Original PyRisk README

*The following is the unmodified README from the original PyRisk repository for reference.*

---

A simple implementation of a variant of the **Risk** board game for python, designed for playing with AIs.

Runs in `python` (2.7 or 3.x) using the `curses` library to display the map (but can be run in pure-console mode).

### Usage

```bash
python pyrisk.py FooAI BarAI*2
```

Use `--help` to see more detailed options, such as multi-game running. The AI loader assumes that `SomeAI` translates to a class `SomeAI` inheriting from `AI` in `ai/some.py`.

### Rules

A minimal version of the **Risk** rules are used:

- Players start with `35 - 5*players` armies.
- At the start of the game, territories are chosen one by one until all are claimed, and then the remaining armies may be deployed one at a time to reinforce claimed territories.
- Each turn, players receive `3 + territories/3` reinforcements, plus a bonus for any complete continents.
- A player may make unlimited attacks per round into adjacent territories (including from freshly-conquered territories).
  - Each combat round, the attacker can attack with up to three armies.
  - Upon victory, a minimum of that combat round's attackers are moved into the target territory.
  - The attacker may cease the attack at the end of any combat round.
  - The defender defends with two armies (unless only one is available).
  - Each attacking and defending army rolls 1d6. The rolls on each side are ordered and compared. The loser of each complete pair is removed, with the defender winning ties.
- At the end of each turn, a player may make one free move.
- Victory is achieved by world domination.

### API

Write a new class extending the `AI` class in `ai/__init__.py`. The methods are documented in that file. At a minimum, the following functions need to be implemented:

- `initial_placement(self, empty, remaining)`: Return an empty territory if any are still listed in `empty`, else an existing territory to reinforce.
- `reinforce(self, available)`: Return a dictionary of territory -> count for the reinforcement step.
- `attack(self)`: Yield `(from, to, attack_strategy, move_strategy)` tuples for each attack you want to make.

The `AI` base class provides objects `game`, `player` and `world` which can be inspected for the current game state. *These are unproxied versions of the main game data structures, so you're trusted not to modify them.*
