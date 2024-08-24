import random
import time
import json

class Nonogram:
    def __init__(self, size: int = 5) -> None:
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.gridString = ""
        self.tipsX = []
        self.tipsY = []
        self.combinedTips = []
        self.score = 0
        self.solution = None
        self.guesses = 0
        self.max_guesses = 5
        self.__pickPuzzle()
        self.__gridToString()

    def __pickPuzzle(self) -> None:
        with open('puzzles/' + str(self.size) + 'x' + str(self.size) + '.json', 'r') as file:
            random.seed(time.time())
            fileData = json.load(file)
            randomNumber = random.randint(0, len(fileData) - 1)

            puzzle = fileData[random.randint(0, len(fileData) - 1)]

            
            self.tipsX = puzzle['tipsX']
            self.tipsY = puzzle['tipsY']
            self.solution = puzzle['solution']
            self.combinedTips = puzzle['combined']

            self.solution = [
              0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0
            ]
            self.combinedTips = [
      0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0
    ]


    #         self.solution = [
    #   0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
    #   1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
    #   0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
    #   1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0
    # ]
    #         self.combinedTips = [
    #   3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
    #   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    #   2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
    #   0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
    #   0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0,
    #   0, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    #   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #   0
    # ]
    
    #         self.solution = [
    #   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
    #   0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1,
    #   1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
    #   1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    # ]
    #         self.combinedTips = [
    #   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0,
    #   0, 0, 0, 0, 0, 1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    #   1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0,
    #   0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0,
    #   0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
    #   0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0,
    #   0, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0, 0,
    #   0, 0, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0,
    #   0
    # ]

    def __gridToString(self) -> str:
        grid_string = ""
        for row in self.grid:
            row_string = ",".join(str(cell) for cell in row)
            grid_string += row_string + "\n"
        self.gridString = grid_string[:-1]

    def print_grid(self):
        for row in self.grid:
            print(" ".join(str(cell) for cell in row))

    def print_solution(self):
        for i in range(0, len(self.solution), self.size):
            row = self.solution[i:i+self.size]
            print(" ".join(str(cell) for cell in row))

    def getSimulationData(self) -> dict:
        return [self.tipsX, self.tipsY, self.combinedTips, self.solution]

    def setGameSet(self, tipsX, tipsY, solution, combinedTips):
        self.tipsX = tipsX
        self.tipsY = tipsY
        self.solution = solution
        self.combinedTips = combinedTips

    def getFlattenedGrid(self) -> str:
        return self.gridString.replace("\n", ",")

    def getFlattenedNumberGrid(self):
        flatList = self.getFlattenedGrid().split(',')
        return [int(cell) for cell in flatList]

    def checkSolution(self) -> bool:
        current_solution = [int(cell) for row in self.grid for cell in row]
        if current_solution == self.solution:
            self.score = 1000
            return True
        else:
            return False

    def checkValidCells(self) -> int:
        for i in range(self.size * self.size):
            if self.gridString[i] != self.solution[i]:
                return i
        if i == self.size * self.size:
            return 1000

    def update_grid(self, x: int, y: int) -> int:
        if 0 <= x < self.size and 0 <= y < self.size:
            solutionIndex = y * self.size + x
            self.grid[y][x] = 1
            self.__gridToString()
            if self.solution[solutionIndex] == 1:
                self.score += 1
                return 0  # Success
            elif self.solution[solutionIndex] == 0:
                self.guesses += 1
                if self.guesses >= self.max_guesses:
                    return 2  # Error (Exceeding max guesses)
                return 1  # Error (Not part of the solution)
            
            pass
        else:
            return -1  # Error (Outside grid)
            

    # def update_grid(self, x: int, y: int) -> int:
    #     if 0 <= x < self.size and 0 <= y < self.size:
    #         solutionIndex = y * self.size + x
    #         if self.solution[solutionIndex] == 1 and self.grid[y][x] == 0:
    #             self.grid[y][x] = 1
    #             self.__gridToString()
    #             self.score += 1
    #             return 0    # Success
    #         elif self.solution[solutionIndex] == 0 and self.grid[y][x] == 1:
    #             return -1    # Error (trying to set a cell that is already set)
    #         else:
    #             self.guesses += 1
    #             if self.guesses >= self.max_guesses:
    #                 return 2  # Error (exceeding max guesses)
    #             return 1  # Error (not part of the solution)
    #     else:
    #         # Click outside grid boundaries (considered an error)
    #         return 1  # Error (not part of the solution)
