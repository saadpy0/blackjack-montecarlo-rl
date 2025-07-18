from blackjack_env import BlackjackEnv
from mc_agent import MonteCarloAgent
import numpy as np
import pickle

if __name__ == "__main__":
    env = BlackjackEnv(n_decks=6, penetration=0.75, counting_limit=2)
    agent = MonteCarloAgent(env, epsilon=0.1, gamma=1.0)
    agent.train(num_episodes=50000)
    # Save Q-table for later evaluation
    with open("q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)
    print("Training complete. Q-table saved to q_table.pkl.") 