from game import Nonogram


def main():
  game = Nonogram()
  simulation_data = game.getSimulationData()
  
  
  print(game.getFlattenedNumberGrid())
  
  # print(simulation_data)
  # print("\n\n")


if __name__ == "__main__":
  main()