import random
import time
import json

class Nonogram:
    def __init__(self, size: int = 5) -> None:
        self.size = size
        self.puzzleId = 0
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.grid_string = ""
        self.tipsX = []
        self.tipsY = []
        self.combinedTips = []
        self.score = 0
        self.solution = None
        self.guesses = 0
        self.max_guesses = 5
        self.__pick_puzzle()
        self.__grid_to_string()

    def __pick_puzzle(self) -> None:
        with open(f'puzzles/{self.size}x{self.size}.json', 'r') as file:
            random.seed(time.time())
            file_data = json.load(file)
            puzzle = random.choice(file_data)

            self.tipsX = puzzle['tipsX']
            self.tipsY = puzzle['tipsY']
            self.solution = puzzle['solution']
            self.combinedTips = puzzle['combined']
            self.puzzleId = puzzle['id']

    def __grid_to_string(self) -> str:
        self.grid_string = "\n".join(",".join(str(cell) for cell in row) for row in self.grid)

    def print_grid(self):
        for row in self.grid:
            print(" ".join(str(cell) for cell in row))

    def print_solution(self):
        for i in range(0, len(self.solution), self.size):
            row = self.solution[i:i+self.size]
            print(" ".join(str(cell) for cell in row))

    def get_simulation_data(self) -> dict:
        return [self.tipsX, self.tipsY, self.combinedTips, self.solution]

    def set_game_set(self, tipsX, tipsY, solution, combinedTips):
        self.tipsX = tipsX
        self.tipsY = tipsY
        self.solution = solution
        self.combinedTips = combinedTips

    def get_flattened_grid(self) -> str:
        return self.grid_string.replace("\n", ",")

    def get_flattened_number_grid(self):
        flat_list = self.get_flattened_grid().split(',')
        return [int(cell) for cell in flat_list]

    def check_solution(self) -> bool:
        current_solution = [int(cell) for row in self.grid for cell in row]
        if current_solution == self.solution:
            self.score = 1000
            return True
        return False

    def update_grid(self, x: int, y: int) -> int:
        if 0 <= x < self.size and 0 <= y < self.size:
            solution_index = y * self.size + x
            self.grid[y][x] = 1
            self.__grid_to_string()
            if self.solution[solution_index] == 1:
                self.score += 1
                return 0  # Success
            elif self.solution[solution_index] == 0:
                self.guesses += 1
                if self.guesses >= self.max_guesses:
                    return 2  # Error (Exceeding max guesses)
                return 1  # Error (Not part of the solution)
        return -1  # Error (Outside grid)
