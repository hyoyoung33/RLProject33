import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from tabulate import tabulate

# For plotting metrics
all_epochs = []
# params
GAMMA = 0.99
LEARNING_COUNT = 20000
TEST_COUNT = 10
TURN_LIMIT= 100
env = gym.make("Taxi-v3")
num_states = env.observation_space.n
num_actions = env.action_space.n
EPS = 0.1

# For Hyperparameter Tuning
OPTI_COUNT = 10
alpha_start = 0.40
alpha_end = 0.50
alpha_opti_rate = (alpha_end - alpha_start) / OPTI_COUNT
alphas = []
alpha = alpha_end
alphas.append(alpha)
for i in range(OPTI_COUNT):
    alpha = alpha - alpha_opti_rate
    round(alpha,2)
    alphas.append(alpha)

class Agent:
    def __init__(self, env):
        self.env = env
        self.episode_reward = 0.0
        self.q_val = np.zeros(num_states * num_actions).reshape(num_states, num_actions).astype(np.float32)

    def learn(self,m,n):
        # one episode learning
        state, _ = self.env.reset()
        epochs = 0
        ALPHA = alphas[n]
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
                epochs += 1
                state = next_state
        return 0.0

    def test(self):
        env_test = gym.make("Taxi-v3", render_mode='ansi')
        state, _ = env_test.reset()
        for t in range(TURN_LIMIT):
            act = np.argmax(self.q_val[state])
            next_state, reward, terminated, truncated, info = env_test.step(act)
            if terminated or truncated:
                return reward
            else:
                state = next_state
        return 0.0  # over limit
def main():
     opti_value = []
     for j in range(OPTI_COUNT):
        agent = Agent(env)
        print("###### LEARNING #####")
        reward_total = 0.0
        for i in tqdm(range(LEARNING_COUNT)):
            reward_total += agent.learn(i,j)
        print("episodes      : {}".format(LEARNING_COUNT))
        print("total reward  : {}".format(reward_total))
        print("average reward: {:.2f}".format(reward_total / LEARNING_COUNT))
        print("Q Value       :{}".format(agent.q_val))
        opti_value.append({"reward_value": reward_total, "alpha": alphas[j]})

        print("###### TEST #####")
        test_reward_total = 0.0
        for i in tqdm(range(TEST_COUNT)):
            test_reward_total += agent.test()
        print("episodes      : {}".format(TEST_COUNT))
        print("total reward  : {}".format(test_reward_total))
        print("average reward: {:.2f}".format(test_reward_total / TEST_COUNT))

     pt_reward_values = [item['reward_value'] for item in opti_value]
     pt_alphas = [item['alpha'] for item in opti_value]

     # Plotting
     #plt.figure(figsize=(8, 5))

     plt.plot(pt_alphas, pt_reward_values, label='LEARNING Total Reward', color='green', marker='o')

     # Adding titles and labels
     plt.title('Alpha Value')
     plt.ylabel('Reward Value')
     plt.xlabel('Alpha')
     # Adding a legend
     plt.legend()
     # Showing the plot
     plt.show()
     print(tabulate(opti_value, headers='keys'))

if __name__ == "__main__":
    main()

