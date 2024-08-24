import neat
import pickle
import os
from game import Nonogram
import glob

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
        checkpoints = glob.glob(os.path.join(self.checkpoint_path, 'neat-checkpoint-*'))
        if checkpoints:
            return max(checkpoints, key=os.path.getctime)
        return None

    def load_checkpoint(self):
        if self.use_best_genome:
            best_genome_path = os.path.join(self.checkpoint_path, 'best_genome.pkl')
            if os.path.exists(best_genome_path):
                print(f"Loading best genome: {best_genome_path}")
                with open(best_genome_path, 'rb') as f:
                    best_genome = pickle.load(f)
                population = neat.Population(self.config)
                population.population[1] = best_genome  # Add the best genome to the population
                return population
            else:
                print("Best genome file not found.")
                return None
        else:
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path:
                print(f"Loading from checkpoint: {checkpoint_path}")
                population = neat.Checkpointer.restore_checkpoint(checkpoint_path)
                return population
            else:
                print("No checkpoint found.")
                return None

    def run_best_genome(self):
        population = self.load_checkpoint()
        if population is None:
            print("No population loaded!")
            return

        best_genome = None
        best_fitness = float('-inf')
        for genome_id, genome in population.population.items():
            if genome.fitness is not None and genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_genome = genome

        if best_genome is None:
            print("No valid genomes found in the population!")
            return

        print(f"Running best genome with fitness: {best_fitness}")

        net = neat.nn.FeedForwardNetwork.create(best_genome, self.config)
        game = Nonogram(self.map_size)
        tipsX, tipsY, combinedTips, solution = game.getSimulationData()
        inputs = self.prepare_inputs(combinedTips, self.map_size)

        for _ in range(1000):  # Limit iterations to avoid infinite loops
            output = net.activate(inputs)

            for i in range(self.map_size * self.map_size):
                x = i % self.map_size
                y = i // self.map_size
                if output[i] > 0.5:
                    game.update_grid(x, y)

            if game.checkSolution():
                print("Solution found by the best genome!")
                print("max_guesses:", game.guesses)
                break

        print("Best genome played the game.")
        game.print_solution()
        self.print_comparison(game.grid, solution)

    def prepare_inputs(self, combinedTips, size):
        assert len(combinedTips) == size * size * 2, f"Expected {size * size * 2} inputs but got {len(combinedTips)}"
        return combinedTips

    def print_comparison(self, game_grid, solution):
        print("\nGame grid (red indicates incorrect cells):")
        
        RESET = '\033[0m'
        RED = '\033[91m'

        solution_index = 0
        for row in game_grid:
            line = ""
            for cell in row:
                if cell != solution[solution_index]:
                    line += f"{RED}{cell}{RESET} "
                else:
                    line += f"{cell} "
                solution_index += 1
            print(line)

if __name__ == "__main__":
    # Example usage:
    # To use the latest checkpoint:
    runner = NEATRunner(5, use_best_genome=False)
    runner.run_best_genome()

    # To use the best_genome.pkl:
    # runner = NEATRunner(5, use_best_genome=True)
    # runner.run_best_genome()