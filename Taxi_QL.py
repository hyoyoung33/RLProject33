import numpy as np
import gymnasium as gym
from tqdm import tqdm
# params
ALPHA = 0.1
GAMMA = 0.99
LEARNING_COUNT = 1000
TEST_COUNT = 1000
EPS = 0.2
TURN_LIMIT= 500
env = gym.make("Taxi-v3",render_mode='ansi')
num_states = env.observation_space.n
num_actions = env.action_space.n

class Agent:
    def __init__(self, env):
        self.env = env
        self.episode_reward = 0.0
        self.q_val = np.zeros(num_states * num_actions).reshape(num_states, num_actions).astype(np.float32)

    def learn(self):
        # one episode learning
        state, _ = self.env.reset()
        penalties, reward, = 0, 0
        progress = []
        for t in range(TURN_LIMIT):
            if np.random.rand() < EPS: # explore
                act = self.env.action_space.sample() # random
            else: # exploit
                act = np.argmax(self.q_val[state])
            next_state, reward, terminated, truncated, info = self.env.step(act)
            q_next_max = np.max(self.q_val[next_state])
            # Q <- Q + a(Q' - Q)
            # <=> Q <- (1-a)Q + a(Q')
            self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act]\
                                 + ALPHA * (reward + GAMMA * q_next_max)

            if terminated or truncated:
                return reward
            else:
                state = next_state
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
        return 0.0 # over limit

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
