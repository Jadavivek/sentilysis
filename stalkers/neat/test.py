import pickle
from .train import NeatStalker

def load_and_run_saved_model(filepath: str):
    with open(filepath, 'rb') as f:
        saved_genome = pickle.load(f)

    config_path = "config"
    neat_stalker = NeatStalker(name="LoadedNeatBot", config_path=config_path, save_dir="results")

    # Create the neural network from the saved genome
    net = neat.nn.FeedForwardNetwork.create(saved_genome, neat_stalker.config)

    inputs = [0.5, 0.5]  
    output = net.activate(inputs)

    action = output.index(max(output))
    if action == 0:
        neat_stalker.scroll("down", 5)
    elif action == 1:
        neat_stalker.click(100, 200)
    elif action == 2:
        neat_stalker.screenshot("loaded_screenshot.png")
    elif action == 3:
        neat_stalker.keypress("Enter")

if __name__ == "__main__":
    load_and_run_saved_model("saves/100.pickle")