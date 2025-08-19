import os 
from os import path
import torch

def analyze_emotions_with_llm(tweets):
    model_path = path.join(path.dirname(__file__), "saves", "transformer_megalodon_multiclass_final.pt")
    try:
        model = torch.load(model_path)
        model.eval()  # Set the model to evaluation mode
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    preprocessed_tweets = [tweet.lower() for tweet in tweets]

    model_inputs = torch.tensor([len(tweet) for tweet in preprocessed_tweets])

    try:
        with torch.no_grad():
            predictions = model(model_inputs)
        print("Predictions:", predictions)
    except Exception as e:
        print(f"Error during inference: {e}")

    return predictions