import numpy as np
import os
import gym
import random
from tqdm import tqdm

total_reward = []


class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.8, gamma=0.9):
        """
        Parameters:
            env: target environment.
            epsilon: Determines the explore/exploit rate of the agent.
            learning_rate: Learning rate of the agent.
            gamma: discount rate of the agent.
        """
        self.env = env

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize qtable
        self.qtable = np.zeros((env.observation_space.n, env.action_space.n))

        self.qvalue_rec = []

    def choose_action(self, state):
        """
        Choose the best action with given state and epsilon.

        Parameters:
            state: A representation of the current state of the environment.
            epsilon: Determines the explore/exploit rate of the agent.

        Returns:
            action: The action to be evaluated.
        """
        # Begin your code
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample()
        else: return np.argmax(self.qtable[state])
        # End your code

    def learn(self, state, action, reward, next_state, done):
        """
        Calculate the new q-value base on the reward and state transformation observered after taking the action.

        Parameters:
            state: The state of the environment before taking the action.
            action: The exacuted action.
            reward: Obtained from the environment after taking the action.
            next_state: The state of the environment after taking the action.
            done: A boolean indicates whether the episode is done.

        Returns:
            None (Don't need to return anything)
        """
        # Begin your code
        estimated_optimal_Q = self.qtable[state, action]
        next_max = np.max(self.qtable[next_state])
        self.qtable[state, action] = (1 - self.learning_rate) * estimated_optimal_Q + self.learning_rate * (reward + self.gamma * next_max)
        # End your code

        # You can add some conditions to decide when to save your table
        # saved = np.load("./Tables/taxi_table.npy")
        if self.qtable[state, action] > 15:
            np.save("./Tables/taxi_table.npy", self.qtable)

    def check_max_Q(self, state):
        """
        - Implement the function calculating the max Q value of given state.
        - Check the max Q value of initial state

        Parameter:
            state: the state to be check.
        Return:
            max_q: the max Q value of given state
        """
        # Begin your code
        return max(self.qtable[state])
        # End your code


def extract_state(ori_state):
        state = []
        
        if ori_state % 4 == 0:
            state.append('R')
        else:
            state.append('G')
        
        ori_state = ori_state // 4

        if ori_state % 5 == 2:
            state.append('Y')
        else:
            state.append('B')
        
        print(f"Initial state:\ntaxi at (2, 2), passenger at {state[1]}, destination at {state[0]}")
        

def train(env):
    """
    Train the agent on the given environment.

    Paramenters:
        env: the given environment.

    Returns:
        None (Don't need to return anything)
    """
    training_agent = Agent(env)
    episode = 3000
    rewards = []
    for ep in tqdm(range(episode)):
        state = env.reset()
        done = False

        count = 0
        while True:
            action = training_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            training_agent.learn(state, action, reward, next_state, done)
            count += reward
            if done:
                rewards.append(count)
                break

            state = next_state
    total_reward.append(rewards)


def test(env):
    """
    Test the agent on the given environment.

    Paramenters:
        env: the given environment.

    Returns:
        None (Don't need to return anything)
    """
    testing_agent = Agent(env)
    testing_agent.qtable = np.load("./Tables/taxi_table.npy")
    rewards = []

    for _ in range(100):
        state = testing_agent.env.reset()
        count = 0
        while True:
            action = np.argmax(testing_agent.qtable[state])
            next_state, reward, done, _ = testing_agent.env.step(action)
            count += reward
            if done == True:
                rewards.append(count)
                break

            state = next_state
    # Please change to the assigned initial state in the Google sheet
    state = 252

    print(f"average reward: {np.mean(rewards)}")
    extract_state(state)
    print(f"max Q:{testing_agent.check_max_Q(state)}")


def seed(seed=20):
    '''
    It is very IMPORTENT to set random seed for reproducibility of your result!
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    '''
    The main funtion
    '''
    # Please change to the assigned seed number in the Google sheet
    SEED = 115

    env = gym.make("Taxi-v3")
    seed(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)
        
    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    # training section:
    for _ in range(5):
        train(env)   
    # testing section:
    test(env)
        
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/taxi_rewards.npy", np.array(total_reward))

    env.close()