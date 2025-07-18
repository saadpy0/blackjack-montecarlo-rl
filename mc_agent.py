import numpy as np
from collections import defaultdict

class MonteCarloAgent:
    def __init__(self, env, epsilon=0.1, gamma=1.0):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns = defaultdict(list)
        self.policy = defaultdict(lambda: np.ones(env.action_space.n) / env.action_space.n)

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False
        while not done:
            probs = self.policy[self._state_to_tuple(state)]
            action = np.random.choice(np.arange(self.env.action_space.n), p=probs)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def _state_to_tuple(self, state):
        # Convert state (with np array) to a hashable tuple
        return (state[0], state[1], state[2], int(state[3][0]))

    def update_policy(self, state):
        best_action = np.argmax(self.Q[state])
        nA = self.env.action_space.n
        self.policy[state] = np.ones(nA) * self.epsilon / nA
        self.policy[state][best_action] += 1.0 - self.epsilon

    def train(self, num_episodes=10000):
        for i in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            visited = set()
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                state_t = self._state_to_tuple(state)
                G = self.gamma * G + reward
                if (state_t, action) not in visited:
                    self.returns[(state_t, action)].append(G)
                    self.Q[state_t][action] = np.mean(self.returns[(state_t, action)])
                    self.update_policy(state_t)
                    visited.add((state_t, action))
            if (i+1) % 1000 == 0:
                print(f"Episode {i+1}/{num_episodes} completed.") 