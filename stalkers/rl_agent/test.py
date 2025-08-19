import numpy as np
from os import path
import os

from stalkers.fitness import get_fitness


def load_q_table(filepath):
    """Load the Q-table from a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Q-table file not found: {filepath}")
    return np.load(filepath)


def perform_actions(
    q_table, page_state, page_prev_state, state_space_size, action_space_size
):
    state = 0
    action = np.argmax(q_table[state])

    reward = get_fitness(page_state, [page_prev_state])
    next_state = (state + 1) % state_space_size
    q_table[state, action] += 0.1 * (
        reward + 0.99 * np.max(q_table[next_state]) - q_table[state, action]
    )

    return int(action)

def get_action(page: str, prev_page: str) -> int:
    save_file = path.join(path.dirname(__file__), "saves", "best.npy")
    q_table = load_q_table(save_file) 
    return perform_actions(q_table, page, prev_page, 10, 5)