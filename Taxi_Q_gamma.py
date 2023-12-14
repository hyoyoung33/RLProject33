import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from tabulate import tabulate

# For plotting metrics
all_epochs = []
# params
ALPHA = 0.47
LEARNING_COUNT = 40000
TEST_COUNT = 10
TURN_LIMIT= 100
env = gym.make("Taxi-v3")
num_states = env.observation_space.n
num_actions = env.action_space.n

# For Hyperparameter Tuning
OPTI_COUNT = 10
gamma_start = 0.90
gamma_end = 0.99
gamma_opti_rate = (gamma_end - gamma_start) / OPTI_COUNT
gammas = []
gamma = gamma_end
gammas.append(gamma)
for i in range(OPTI_COUNT):
    gamma = gamma - gamma_opti_rate
    round(gamma,2)
    gammas.append(gamma)

EPS = 0.1
class Agent:
    def __init__(self, env):
        self.env = env
        self.episode_reward = 0.0
        self.q_val = np.zeros(num_states * num_actions).reshape(num_states, num_actions).astype(np.float32)

    def learn(self,m,n):
        # one episode learning
        state, _ = self.env.reset()
        epochs = 0
        GAMMA = gammas[n]
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
        opti_value.append({"reward_value": reward_total, "gamma": gammas[j]})

        print("###### TEST #####")
        test_reward_total = 0.0
        for i in tqdm(range(TEST_COUNT)):
            test_reward_total += agent.test()
        print("episodes      : {}".format(TEST_COUNT))
        print("total reward  : {}".format(test_reward_total))
        print("average reward: {:.2f}".format(test_reward_total / TEST_COUNT))

     pt_reward_values = [item['reward_value'] for item in opti_value]
     pt_gammas = [item['gamma'] for item in opti_value]

     # Plotting
     #plt.figure(figsize=(8, 5))

     plt.plot(pt_gammas, pt_reward_values, label='LEARNING Total Reward', color='red', marker='o')

     # Adding titles and labels
     plt.title('Gamma Value')
     plt.ylabel('Reward Value')
     plt.xlabel('Gamma')
     # Adding a legend
     plt.legend()
     # Showing the plot
     plt.show()
     print(tabulate(opti_value, headers='keys'))

if __name__ == "__main__":
    main()

