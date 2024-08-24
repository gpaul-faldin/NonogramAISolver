import neat
from neat.checkpoint import Checkpointer
import pickle
import os
import re
from game import Nonogram

class NEATTrainer:
    def __init__(self, map_size=5, fitness_threshold=1000, consecutive_threshold=10, config_path='neat-config.txt', checkpoint_path='logs/'):
        self.map_size = map_size
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path
        )
        self.population = None
        self.generation = 0
        self.fitness_threshold = fitness_threshold
        self.consecutive_threshold = consecutive_threshold
        self.checkpointer = Checkpointer(generation_interval=5, filename_prefix=f'{checkpoint_path}neat-checkpoint-')
        self.best_genome = None
        self.threshold_count = 0

    def load_checkpoint(self):
        checkpoint_files = [f for f in os.listdir(self.checkpoint_path) if f.startswith('neat-checkpoint-')]

        if checkpoint_files:
            # Custom sorting key: extract the number from the filename
            def checkpoint_number(filename):
                return int(re.search(r'-(\d+)$', filename).group(1))

            latest_checkpoint = max(checkpoint_files, key=checkpoint_number)
            checkpoint_path = os.path.join(self.checkpoint_path, latest_checkpoint)
            print(f"Resuming from checkpoint: {checkpoint_path}")
            self.population = neat.Checkpointer.restore_checkpoint(checkpoint_path)
            self.generation = checkpoint_number(latest_checkpoint)
            print(f"Resumed from checkpoint at generation {self.generation}")
        else:
            print("No checkpoint found. Starting a new population.")
            self.population = neat.Population(self.config)

        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(neat.StatisticsReporter())
        self.population.add_reporter(self.checkpointer)

    def evaluate_genomes(self, genomes, config):
        TrainingGame = Nonogram(self.map_size)
        print(f"Puzzle ID: {TrainingGame.puzzleId}")
        results = []
        log = True

        for i, (genome_id, genome) in enumerate(genomes):
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            game = Nonogram(self.map_size)
            game.set_game_set(TrainingGame.tipsX, TrainingGame.tipsY, TrainingGame.solution, TrainingGame.combinedTips)
            wrong_guess_cells = []
            max_guess_stop = False
            cell_guessed = 0

            for x in range(self.map_size):
                for y in range(self.map_size):
                    if max_guess_stop:
                        break

                    input_data = game.combinedTips + game.get_flattened_number_grid() + [x, y]
                    if i == 0 and log:
                        print(f"Input data: {input_data}")
                    output = net.activate(input_data)

                    if output[0] > 0.5:
                        cell_guessed += 1
                        res = game.update_grid(x, y)
                        if res == 2:
                            max_guess_stop = True
                        elif res == 1:
                            wrong_guess_cells.append((x, y))

            if not max_guess_stop and wrong_guess_cells:
                for cell in wrong_guess_cells:
                    if max_guess_stop:
                        break

                    input_data = game.combinedTips + game.get_flattened_number_grid() + [cell[0] / self.map_size, cell[1] / self.map_size]
                    output = net.activate(input_data)

                    if output[0] > 0.5:
                        res = game.update_grid(cell[0], cell[1])
                        if res == 2:
                            max_guess_stop = True

            game.check_solution()

            # Penalize for stopping early due to max_guess_stop
            penalty = self.map_size * self.map_size - cell_guessed if max_guess_stop else 0

            # Adjust fitness calculation
            fitness = game.score - (game.guesses - game.max_guesses - 1) - penalty
            genome.fitness = max(0, fitness)
            results.append((genome_id, genome, genome.fitness))

    def save_best_genome(self):
        if self.best_genome:
            with open(os.path.join(self.checkpoint_path, 'best_genome.pkl'), 'wb') as f:
                pickle.dump(self.best_genome, f)
            print(f"Best genome saved with fitness: {self.best_genome.fitness}")

    def run_training(self, generations=100, resume=False):
        if resume:
            self.load_checkpoint()
        else:
            self.population = neat.Population(self.config)
            self.population.add_reporter(self.checkpointer)
            self.population.add_reporter(neat.StdOutReporter(True))
            self.population.add_reporter(neat.StatisticsReporter())
            print("Started a new random population")

        self.population.add_reporter(self.checkpointer)

        for _ in range(generations - self.generation):
            self.generation += 1
            winner = self.population.run(self.evaluate_genomes, 1)

            if winner.fitness >= self.fitness_threshold:
                self.threshold_count += 1
                print(f"Fitness threshold met for {self.threshold_count} consecutive generations!")
                if self.threshold_count >= self.consecutive_threshold:
                    print(f"Fitness threshold met for {self.consecutive_threshold} consecutive generations at generation {self.generation}!")
                    self.best_genome = winner
                    self.save_best_genome()
                    break
            else:
                self.threshold_count = 0

            if not self.best_genome or winner.fitness > self.best_genome.fitness:
                self.best_genome = winner
                self.save_best_genome()

        if self.generation == generations:
            print(f"Reached maximum generations. Best fitness: {self.best_genome.fitness}")
            self.save_best_genome()

def main():
    trainer = NEATTrainer(5)
    trainer.run_training(generations=1000, resume=False)

if __name__ == "__main__":
    main()
