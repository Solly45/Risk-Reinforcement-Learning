import matplotlib.pyplot as plt
import json
import os

class GameStatistics:

    def __init__(self):
        self.stats = {}

    def record_event(self, event_name, data):
        if event_name not in self.stats:
            self.stats[event_name] = []
        self.stats[event_name].append(data)

    def save_statistics(self, filename):
        if not self.stats:
            print("No game statistics available to save.")
            return
        import json
        with open(filename, 'w') as f:
            json.dump(self.stats, f, indent=4)
        print(f"Game statistics saved to {filename}")
    def __init__(self):
        self.turn_counts = []  # Track number of turns per game
        self.win_counts = {}   # Track win counts by AI name
        self.total_games = 0

    def record_game(self, winner_name, turns):
        """Update statistics after a game."""
        self.total_games += 1
        self.turn_counts.append(turns)
        if winner_name not in self.win_counts:
            self.win_counts[winner_name] = 0
        self.win_counts[winner_name] += 1

    def save_statistics(self, filepath="game_stats.json"):
        """Save statistics to a file."""
        with open(filepath, "w") as f:
            json.dump({
                "turn_counts": self.turn_counts,
                "win_counts": self.win_counts,
                "total_games": self.total_games
            }, f)

    def plot_statistics(self, model_name):
        """Plot statistics including turns per game and win percentage."""
        # Plot turns per game
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.turn_counts) + 1), self.turn_counts, label="Turns per Game")
        plt.title("Number of Turns per Game")
        plt.xlabel("Game")
        plt.ylabel("Number of Turns")
        plt.legend()

        # Calculate win percentage every 10 games
        games_played = list(range(10, self.total_games + 1, 10))
        win_percentages = [
        (self.win_counts.get(model_name, 0) / i) * 100
        for i in games_played
    ]

        # Plot win percentage over games
        plt.subplot(1, 2, 2)
        plt.plot(games_played, win_percentages, label=f"{model_name} Win Percentage")
        plt.title("Win Percentage (Per 10 Games)")
        plt.xlabel("Number of Games")
        plt.ylabel("Win Percentage")
        plt.legend()

        plt.tight_layout()
        plt.show()
