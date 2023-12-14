import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from tabulate import tabulate

# params
ALPHA = 0.47
GAMMA = 0.97
LEARNING_COUNT = 50000
TEST_COUNT = 100
TURN_LIMIT= 100
env = gym.make("Taxi-v3")
num_states = env.observation_space.n
num_actions = env.action_space.n

epsilon_start = 1.0
interval = LEARNING_COUNT // 10
pct_decay = interval / LEARNING_COUNT
epsilons = []
epsilon = epsilon_start
for s in range(LEARNING_COUNT):
    if(s % interval == 0 and s !=0):
        epsilon = epsilon - pct_decay
        epsilons.append(epsilon)
    else:
        epsilons.append(epsilon)
# Plotting the epsilons
plt.plot(epsilons)
plt.ylabel('epsilon')
plt.xlabel('LEARNING_COUNT')
plt.title("Interval Decay")
plt.show()
plt.close()

p_steps = []
class Agent:
    def __init__(self, env):
        self.steps = 0
        self.env = env
        self.episode_reward = 0.0
        self.q_val = np.zeros(num_states * num_actions).reshape(num_states, num_actions).astype(np.float32)

    def learn(self,m):
        # one episode learning
        state, _ = self.env.reset()
        #self.steps = 0
        for t in range(TURN_LIMIT):
            #print(epsilons[m])
            if np.random.rand() < epsilons[m]: # explore
            #if np.random.rand() < 0.1:  # explore
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
                if terminated:
                    p_steps.append(self.steps)
                return reward
            else:
                state = next_state
                self.steps += 1
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
        reward_total += agent.learn(i)
    print("episodes      : {}".format(LEARNING_COUNT))
    print("total reward  : {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / LEARNING_COUNT))
    print("Q Value       :{}".format(agent.q_val))
    plt.plot(p_steps)
    plt.ylabel('steps')
    plt.xlabel('LEARNING_COUNT')
    plt.title("Interval Decay")
    plt.show()

    print("###### TEST #####")
    reward_total = 0.0
    for i in tqdm(range(TEST_COUNT)):
        reward_total += agent.test()
    print("episodes      : {}".format(TEST_COUNT))
    print("total reward  : {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / TEST_COUNT))


if __name__ == "__main__":
    main()
