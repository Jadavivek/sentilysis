import numpy as np
import random

from stalkers.stalker import Stalker
from stalkers.fitness import get_fitness

class SarsaAgent:
    def __init__(self, state_space_size, action_space_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        """Initialize the SARSA agent."""
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((state_space_size, action_space_size))

    def choose_action(self, state):
        """Choose an action using an epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state, next_action):
        """Update the Q-value using the SARSA update rule."""
        td_target = reward + self.gamma * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def train(self, environment, episodes):
        """Train the agent in the given environment."""
        for episode in range(episodes):
            state = environment.reset()
            action = self.choose_action(state)

            done = False
            while not done:
                next_state, reward, done, _ = environment.step(action)
                next_action = self.choose_action(next_state)

                self.update(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

    def get_policy(self):
        """Extract the policy from the Q-table."""
        return np.argmax(self.q_table, axis=1)

class SarsaStalker(Stalker):
    def __init__(self, name: str, state_space_size: int, action_space_size: int, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(name)
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((state_space_size, action_space_size))

    def choose_action(self, state):
        """Choose an action using an epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state, next_action):
        """Update the Q-value using the SARSA update rule."""
        td_target = reward + self.gamma * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def train(self, environment, episodes):
        """Train the agent in the given environment."""
        for episode in range(episodes):
            state = environment.reset()
            action = self.choose_action(state)

            done = False
            while not done:
                next_state, reward, done, _ = environment.step(action)
                next_action = self.choose_action(next_state)

                self.update(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

    def get_policy(self):
        """Extract the policy from the Q-table."""
        return np.argmax(self.q_table, axis=1)

    def save_model(self, filename: str):
        """Save the trained Q-table model to the saves/ folder."""
        save_path = f"saves/{filename}"
        np.save(save_path, self.q_table)
        print(f"Model saved to {save_path}.")

    def scroll(self, direction: str, amount: int) -> None:
        """Simulate scrolling in the browser."""
        if direction == "down":
            self.page.evaluate(f"window.scrollBy(0, {amount})")
        elif direction == "up":
            self.page.evaluate(f"window.scrollBy(0, -{amount})")
        print(f"Scrolled {direction} by {amount} units.")

    def click(self, x: int, y: int) -> None:
        """Simulate a mouse click at the specified coordinates."""
        self.page.mouse.click(x, y)
        print(f"Clicked at position ({x}, {y}).")

    def screenshot(self, filename: str) -> None:
        """Take a screenshot of the current page."""
        self.page.screenshot(path=filename)
        print(f"Screenshot saved as {filename}.")

    def keypress(self, key: str) -> None:
        """Simulate a keypress event."""
        self.page.keyboard.press(key)
        print(f"Key '{key}' pressed.")

class Environment:
    def __init__(self, stalker):
        """Initialize the environment with a stalker instance."""
        self.stalker = stalker
        self.state = 0
        self.done = False

    def reset(self):
        """Reset the environment to its initial state."""
        self.state = 0
        self.done = False
        return self.state

    def step(self, action):
        """Perform an action and return the next state, reward, and done flag."""
        # Perform the action using the stalker instance
        if action == 0:
            self.stalker.scroll("down", 5)
        elif action == 1:
            self.stalker.click(100, 200)
        elif action == 2:
            self.stalker.screenshot("screenshot.png")
        elif action == 3:
            self.stalker.keypress("Enter")

        # Calculate the reward using the get_fitness function
        reward = get_fitness(self.stalker.page_text(), [self.stalker.get_prev_page()])

        # Simulate state transition and termination condition
        self.state = (self.state + action) % 100  # Example state transition
        self.done = self.state > 90  # Example termination condition

        return self.state, reward, self.done, {}

    def get_state(self):
        """Return the current state of the environment."""
        return self.state

    def is_done(self):
        """Check if the environment is in a terminal state."""
        return self.done

if __name__ == "__main__":
    # Initialize the SarsaStalker instance
    sarsa_stalker = SarsaStalker(
        name="SarsaBot",
        state_space_size=100,  # Example size
        action_space_size=4,  # Example size
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )

    # Initialize the environment with the SarsaStalker instance
    environment = Environment(sarsa_stalker)

    # Train the SarsaStalker
    episodes = 100  # Number of episodes to train
    sarsa_stalker.train(environment, episodes)

    # Save the trained model
    sarsa_stalker.save_model("sarsa_model.npy")

    # Print the learned policy
    policy = sarsa_stalker.get_policy()
    print("Learned Policy:", policy)