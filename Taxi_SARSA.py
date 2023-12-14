import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
# params
ALPHA = 0.1
GAMMA = 0.99
LEARNING_COUNT = 50000
TEST_COUNT = 1000
EPS = 0.2
TURN_LIMIT= 500
env = gym.make("Taxi-v3",render_mode='ansi')
num_states = env.observation_space.n
num_actions = env.action_space.n

def action_epsilon_greedy(q):
    if np.random.rand() > EPS:
        return np.argmax(q)
    return env.action_space.sample()

class Agent:
    def __init__(self, i_env):
        self.env = i_env
        self.episode_reward = 0.0
        self.q_val = np.zeros(num_states * num_actions).reshape(num_states, num_actions).astype(np.float32)

    def learn(self):
        # one episode learning
        state, _ = self.env.reset()
        self.env.render()
        act = action_epsilon_greedy(self.q_val[state])
        for t in range(TURN_LIMIT):
            next_state, reward, terminated, truncated, info = self.env.step(act)
            next_act = action_epsilon_greedy(self.q_val[next_state])
            self.q_val[state][act] = self.q_val[state][act] + ALPHA * (
                        reward + GAMMA * self.q_val[next_state][next_act] - self.q_val[state][act])

            if terminated or truncated:
                return reward
            else:
                state = next_state
                act = next_act

        return 0.0

    def test(self):
        state, _ = self.env.reset()
        for t in range(TURN_LIMIT):
            act = np.argmax(self.q_val[state])
            next_state, reward, terminated, truncated, info = self.env.step(act)
            if terminated or truncated:
                return reward
            else:
                state = next_state
        return 0.0  # over limit


def main():
    agent = Agent(env)
    print("###### LEARNING #####")
    reward_total = 0.0
    for i in tqdm(range(LEARNING_COUNT)):
        reward_total += agent.learn()
    print("episodes      : {}".format(LEARNING_COUNT))
    print("total reward  : {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / LEARNING_COUNT))
    #print("Q Value       :{}".format(agent.q_val))

    print("###### TEST #####")
    reward_total = 0.0
    for i in tqdm(range(TEST_COUNT)):
        reward_total += agent.test()
    print("episodes      : {}".format(TEST_COUNT))
    print("total reward  : {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / TEST_COUNT))


if __name__ == "__main__":
    main()
