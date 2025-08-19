import math
import numpy as np
from collections import Counter

def calculate_entropy(text: str):
    if not text:
        return 0

    frequency = Counter(str(text))
    total_characters = len(text)

    entropy = -sum((int(freq) / int(total_characters)) * math.log2(int(freq) / int(total_characters)) for freq in frequency.values())
    return entropy

def calculate_cross_entropy(agent_text, reference_text):
    agent_freq = Counter(agent_text)
    reference_freq = Counter(reference_text)

    total_agent = len(agent_text)
    total_reference = len(reference_text)

    cross_entropy = 0
    for char in agent_freq:
        p_agent = agent_freq[char] / total_agent
        p_reference = reference_freq.get(char, 1 / (total_reference + 1)) / total_reference
        cross_entropy += p_agent * math.log2(p_reference)

    return -cross_entropy

def calculate_kl_divergence(agent_text, reference_text):
    agent_freq = Counter(agent_text)
    reference_freq = Counter(reference_text)

    total_agent = len(agent_text)
    total_reference = len(reference_text)

    kl_divergence = 0
    for char in agent_freq:
        p_agent = agent_freq[char] / total_agent
        p_reference = reference_freq.get(char, 1 / (total_reference + 1)) / total_reference
        kl_divergence += p_agent * math.log2(p_agent / p_reference)

    return kl_divergence

def get_fitness(agent_text, reference_texts):
    if not reference_texts:
        raise ValueError("Reference texts cannot be empty.")

    agent_entropy = calculate_entropy(agent_text)

    reference_entropies = [calculate_entropy(ref) for ref in reference_texts]
    average_reference_entropy = np.mean(reference_entropies)

    cross_entropies = [calculate_cross_entropy(agent_text, ref) for ref in reference_texts]
    average_cross_entropy = np.mean(cross_entropies)

    kl_divergences = [calculate_kl_divergence(agent_text, ref) for ref in reference_texts]
    average_kl_divergence = np.mean(kl_divergences)

    # Fitness is inversely proportional to the difference in entropy, cross-entropy, and KL divergence
    fitness = 1 / (1 + abs(agent_entropy - average_reference_entropy) + average_cross_entropy + average_kl_divergence)

    # Adjust fitness based on the agent's entropy 
    # and the average reference entropy
    adjustment_factor = math.exp(-math.sqrt(abs(agent_entropy - average_reference_entropy))) # getting the magnitude in the imaginary plane
    adjustment_factor *= math.sin(average_cross_entropy) ** 2 + math.cos(agent_entropy) ** 2 # getting the angle in the imaginary plane
    adjustment_factor += np.tanh(average_kl_divergence) * np.log1p(agent_entropy) # getting the absolute probability of the fitness 
    fitness *= adjustment_factor

    return fitness