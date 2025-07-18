from blackjack_env import BlackjackEnv
import numpy as np
import pickle

def state_to_tuple(state):
    return (state[0], state[1], state[2], int(state[3][0]))

if __name__ == "__main__":
    env = BlackjackEnv(n_decks=6, penetration=0.75, counting_limit=2)
    with open("q_table.pkl", "rb") as f:
        Q = pickle.load(f)
    n_episodes = 10000
    results = {"win": 0, "loss": 0, "draw": 0}
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            state_t = state_to_tuple(state)
            if state_t in Q:
                action = np.argmax(Q[state_t])
            else:
                action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
        if reward == 1:
            results["win"] += 1
        elif reward == -1:
            results["loss"] += 1
        else:
            results["draw"] += 1
    print(f"Results over {n_episodes} episodes:")
    print(f"Wins: {results['win']}")
    print(f"Losses: {results['loss']}")
    print(f"Draws: {results['draw']}") 