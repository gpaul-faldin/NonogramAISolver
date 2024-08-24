import neat
from neat.checkpoint import Checkpointer
import pickle
import os
import re
from game import Nonogram

class NEATTrainer:
    def __init__(self, map_size=5, fitness_threshold=1300, consecutive_threshold=10, config_path='neat-config.txt', checkpoint_path='logs/'):
        self.map_size = map_size
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.config = None
        self.population = None
        self.generation = 0
        self.fitness_threshold = fitness_threshold
        self.consecutive_threshold = consecutive_threshold
        self.checkpointer = Checkpointer(generation_interval=5, filename_prefix=f'{checkpoint_path}neat-checkpoint-')
        self.best_genome = None
        self.threshold_count = 0
        self.setup_neat()

    def setup_neat(self):
        num_inputs = self.map_size * self.map_size * 2  # Row clues + Column clues
        num_outputs = self.map_size * self.map_size  # Grid cells

        # Create a NEAT configuration dynamically
        with open(self.config_path, 'w') as config_file:
            config_file.write(f"""
            [NEAT]
            fitness_criterion = max
            fitness_threshold = 1300
            pop_size = 1000
            reset_on_extinction = False

            [DefaultGenome]
            num_inputs = {num_inputs}
            num_outputs = {num_outputs}
            activation_default = tanh
            activation_mutate_rate = 0.2
            activation_options = tanh
            aggregation_default = sum
            aggregation_mutate_rate = 0.2
            aggregation_options = sum
            bias_init_mean = 0.0
            bias_init_stdev = 1.0
            bias_max_value = 5.0
            bias_min_value = -5.0
            bias_mutate_power = 0.8
            bias_mutate_rate = 0.3
            bias_replace_rate = 0.1
            compatibility_disjoint_coefficient = 1.5
            compatibility_weight_coefficient = 0.7
            conn_add_prob = 0.3
            conn_delete_prob = 0.2
            enabled_default = True
            enabled_mutate_rate = 0.05
            feed_forward = True
            initial_connection = full_direct
            node_add_prob = 0.2
            node_delete_prob = 0.1
            num_hidden = 0
            response_init_mean = 1.0
            response_init_stdev = 0.5
            response_max_value = 5.0
            response_min_value = -5.0
            response_mutate_power = 0.5
            response_mutate_rate = 0.2
            response_replace_rate = 0.1
            weight_init_mean = 0.0
            weight_init_stdev = 1.0
            weight_max_value = 5.0
            weight_min_value = -5.0
            weight_mutate_power = 0.8
            weight_mutate_rate = 0.3
            weight_replace_rate = 0.1

            [DefaultSpeciesSet]
            compatibility_threshold = 3

            [DefaultStagnation]
            max_stagnation = 20
            species_elitism = 3
            species_fitness_func = max

            [DefaultReproduction]
            elitism = 10
            """)
        self.config = neat.Config(
            neat.DefaultGenome, 
            neat.DefaultReproduction, 
            neat.DefaultSpeciesSet, 
            neat.DefaultStagnation, 
            self.config_path
        )
        self.population = None

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

    def prepare_inputs(self, combinedTips, size):
        assert len(combinedTips) == size * size * 2, f"Expected {size * size * 2} inputs but got {len(combinedTips)}"
        return combinedTips

    def prepare_outputs(self, solution, size):
        outputs = [0] * (size * size)
        for i, value in enumerate(solution):
            outputs[i] = value
        return outputs

    def evaluate_genomes(self, genomes, config):
        results = []
        TrainingGame = Nonogram(self.map_size)

        for genome_id, genome in genomes:
            try:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                game = Nonogram(self.map_size)
                game.setGameSet(TrainingGame.tipsX, TrainingGame.tipsY, TrainingGame.solution, TrainingGame.combinedTips)

                tipsX, tipsY, combinedTips, solution = game.getSimulationData()
                inputs = self.prepare_inputs(combinedTips, self.map_size)

                output = net.activate(inputs)

                fitness = 0
                wrong_guesses = 0
                total_moves = 0
                correct_moves = 0

                for i in range(self.map_size * self.map_size):
                    x = i % self.map_size
                    y = i // self.map_size
                    value = output[i]

                    if value > 0.5:  # AI decides to make a move
                        total_moves += 1
                        update_result = game.update_grid(x, y)

                        if update_result == 0:  # Correct move
                            fitness += 10  # Higher reward for correct moves
                            correct_moves += 1
                        elif update_result == -1:  # Already filled
                            fitness -= 1  # Small penalty for redundant moves
                        elif update_result == 1:  # Wrong move
                            fitness -= 5  # Significant penalty for wrong moves
                            wrong_guesses += 1

                        if wrong_guesses >= game.max_guesses:
                            break  # Stop if max wrong guesses reached

                # Updated fitness components
                if game.checkSolution():
                    fitness = self.fitness_threshold
                else:
                    # Reward for partial solutions
                    correct_cells = sum(1 for i in range(self.map_size * self.map_size) 
                                        if game.grid[i // self.map_size][i % self.map_size] == solution[i])
                    fitness += correct_cells * 5

                # Granular penalty for wrong guesses
                guess_penalty = (wrong_guesses / game.max_guesses) * 100
                fitness -= guess_penalty

            except Exception as e:
                print(f"Error evaluating genome {genome_id}: {str(e)}")
                fitness = 0

            genome.fitness = max(0, fitness)  # Ensure fitness is non-negative
            results.append((genome_id, genome, genome.fitness))

        return results

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
