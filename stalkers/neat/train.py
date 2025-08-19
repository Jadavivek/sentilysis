import neat
import os
import json

from ..stalker import Stalker
from ..fitness import get_fitness

class NeatStalker(Stalker):
    def __init__(self, name: str, config_path: str, save_dir: str):
        super().__init__(name)
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(neat.StatisticsReporter())
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def scroll(self, direction: str, amount: int) -> None:
        super().scroll(direction, amount)

    def click(self, x: int, y: int) -> None:
        super().click(x, y)

    def screenshot(self, filename: str) -> None:
        super().screenshot(filename)

    def keypress(self, key: str) -> None:
        super().keypress(key)

    def save_run(self, generation: int, genome):
        save_path = os.path.join(self.save_dir, f"generation_{generation}.json")
        with open(save_path, "w") as f:
            json.dump({"genome_id": genome.key, "fitness": genome.fitness}, f)
        print(f"Saved generation {generation} data to {save_path}.")

    def evaluate_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            fitness = 0

            # Example: Evaluate fitness based on actions
            for _ in range(10):
                output = net.activate([0.5, 0.5])  # Example input
                action = output.index(max(output))

                if action == 0:
                    self.scroll("down", 5)
                elif action == 1:
                    self.click(100, 200)
                elif action == 2:
                    self.screenshot("neat_screenshot.png")
                elif action == 3:
                    self.keypress("Enter")

                fitness += get_fitness(self.page, [self.get_prev_page()])  # Example fitness increment

            genome.fitness = fitness

    def run(self, generations: int):
        for generation in range(generations):
            self.population.run(self.evaluate_genomes, 100)
            best_genome = max(
                self.population.population.values(), key=lambda g: g.fitness
            )
            self.save_run(generation, best_genome)


def neat_algorithm_loop():
    neat_stalker = NeatStalker(
        name="NeatBot", config_path="config", save_dir="results"
    )

    # Example loop for the Neat algorithm
    for i in range(100):
        print(f"Iteration {i + 1} of the Neat algorithm loop")
        neat_stalker.run(generations=10)
        print("---")


if __name__ == "__main__":
    neat_algorithm_loop()
