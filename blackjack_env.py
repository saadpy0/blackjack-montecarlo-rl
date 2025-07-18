import gym
from gym import spaces
import numpy as np
import random

class BlackjackEnv(gym.Env):
    """
    Custom Blackjack environment with partial deck knowledge, card counting limits, and realistic shuffling.
    """
    def __init__(self, n_decks=6, penetration=0.75, counting_limit=2):
        super(BlackjackEnv, self).__init__()
        self.n_decks = n_decks
        self.penetration = penetration  # Fraction of deck before reshuffle
        self.counting_limit = counting_limit  # How much info agent can get about deck
        self.action_space = spaces.Discrete(2)  # 0: stick, 1: hit
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # player sum
            spaces.Discrete(11),  # dealer card
            spaces.Discrete(2),   # usable ace
            spaces.Box(low=-counting_limit, high=counting_limit, shape=(1,), dtype=np.int8)  # partial count info
        ))
        self._init_deck()
        self.reset()

    def _init_deck(self):
        self.deck = [1,2,3,4,5,6,7,8,9,10,10,10,10] * 4 * self.n_decks
        random.shuffle(self.deck)
        self.initial_deck_size = len(self.deck)

    def draw_card(self):
        if len(self.deck) < (1 - self.penetration) * self.initial_deck_size:
            self._init_deck()
        return self.deck.pop()

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def get_partial_count(self):
        # Give agent a noisy, limited version of the true count
        true_count = sum(1 if c in [2,3,4,5,6] else -1 if c in [10,1] else 0 for c in self.deck)
        noise = np.random.randint(-1, 2)
        partial = np.clip(true_count + noise, -self.counting_limit, self.counting_limit)
        return np.array([partial], dtype=np.int8)

    def reset(self):
        self.player = self.draw_hand()
        self.dealer = self.draw_hand()
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        return (self._sum_hand(self.player), self.dealer[0], self._usable_ace(self.player), self.get_partial_count())

    def _usable_ace(self, hand):
        return int(1 in hand and sum(hand) + 10 <= 21)

    def _sum_hand(self, hand):
        s = sum(hand)
        if 1 in hand and s + 10 <= 21:
            return s + 10
        return s

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit
            self.player.append(self.draw_card())
            if self._sum_hand(self.player) > 21:
                self.done = True
                return self._get_obs(), -1, True, {}
            else:
                return self._get_obs(), 0, False, {}
        else:  # stick
            while self._sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = self._cmp(self._sum_hand(self.player), self._sum_hand(self.dealer))
            self.done = True
            return self._get_obs(), reward, True, {}

    def _cmp(self, player, dealer):
        if player > 21:
            return -1
        elif dealer > 21 or player > dealer:
            return 1
        elif player == dealer:
            return 0
        else:
            return -1 