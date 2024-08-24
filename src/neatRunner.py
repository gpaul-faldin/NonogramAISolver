import neat
import pickle
import os
import re
from game import Nonogram

class NEATRunner:
    def __init__(self, map_size=5, checkpoint_path='logs/', config_path='neat-config.txt', use_best_genome=False):
        self.map_size = map_size
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.use_best_genome = use_best_genome
        self.config = neat.Config(
            neat.DefaultGenome, 
            neat.DefaultReproduction, 
            neat.DefaultSpeciesSet, 
            neat.DefaultStagnation, 
            self.config_path
        )

    def find_latest_checkpoint(self):
        checkpoint_files = [f for f in os.listdir(self.checkpoint_path) if f.startswith('neat-checkpoint-')]
        if checkpoint_files:
            def checkpoint_number(filename):
                return int(re.search(r'-(\d+)$', filename).group(1))
            return max(checkpoint_files, key=checkpoint_number)
        return None

    def load_checkpoint(self):
        if self.use_best_genome:
            best_genome_path = os.path.join(self.checkpoint_path, 'best_genome.pkl')
            if os.path.exists(best_genome_path):
                print(f"Loading best genome: {best_genome_path}")
                with open(best_genome_path, 'rb') as f:
                    best_genome = pickle.load(f)
                return best_genome
            else:
                print("Best genome file not found.")
                return None
        else:
            checkpoint_file = self.find_latest_checkpoint()
            if checkpoint_file:
                checkpoint_path = os.path.join(self.checkpoint_path, checkpoint_file)
                print(f"Loading from checkpoint: {checkpoint_path}")
                population = neat.Checkpointer.restore_checkpoint(checkpoint_path)
                # Filter out genomes with None fitness
                valid_genomes = [g for g in population.population.values() if g.fitness is not None]
                if valid_genomes:
                    return max(valid_genomes, key=lambda g: g.fitness)
                else:
                    print("No valid genomes with fitness found.")
                    return None
            else:
                print("No checkpoint found.")
                return None


    def run_best_genome(self):
        best_genome = self.load_checkpoint()
        if best_genome is None:
            print("No valid genome loaded!")
            return

        print(f"Running best genome with fitness: {best_genome.fitness}")

        net = neat.nn.FeedForwardNetwork.create(best_genome, self.config)
        game = Nonogram(self.map_size)
        game.set_game_set(game.tipsX, game.tipsY, game.solution, game.combinedTips)

        for x in range(self.map_size):
            for y in range(self.map_size):
                input_data = game.combinedTips + game.get_flattened_number_grid() + [x / self.map_size, y / self.map_size]
                output = net.activate(input_data)
                if output[0] > 0.5:
                    game.update_grid(x, y)

        game.check_solution()
        print(f"Final score: {game.score}")
        print(f"Number of guesses: {game.guesses}")
        print("Best genome played the game.")
        game.print_solution()
        self.print_comparison(game.grid, game.solution)

    def print_comparison(self, game_grid, solution):
        print("\nGame grid (red indicates incorrect cells):")
        
        RESET = '\033[0m'
        RED = '\033[91m'

        for i, row in enumerate(game_grid):
            line = ""
            for j, cell in enumerate(row):
                if cell != solution[i * self.map_size + j]:
                    line += f"{RED}{cell}{RESET} "
                else:
                    line += f"{cell} "
            print(line)

if __name__ == "__main__":
    # Example usage:
    # To use the latest checkpoint:
    runner = NEATRunner(5, use_best_genome=False)
    runner.run_best_genome()

    # To use the best_genome.pkl:
    # runner = NEATRunner(5, use_best_genome=True)
    # runner.run_best_genome()