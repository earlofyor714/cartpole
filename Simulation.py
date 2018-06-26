import gym
import numpy as np
import os
import csv

from Agent import Agent


class Simulator(object):
    def __init__(self, env, display=True, log_metrics=False, filename="sim"):
        self.env = env
        self.agent = Agent(env.observation_space.shape[0], env.action_space.n)
        self.testing = False
        self.log_metrics = log_metrics
        self.display = display

        if self.log_metrics:
            self.log_filename = os.path.join("logs", filename+"_cartpole.csv")
            self.log_fields = ['episode', 'testing', 'net_reward', 'epsilon', 'gamma', 'alpha']
            self.log_file = open(self.log_filename, 'w', newline='')
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=self.log_fields)
            self.log_writer.writeheader()

    def log_trial(self, episode, net_reward):
        if self.log_metrics:
            self.log_writer.writerow({
                'episode': episode,
                'testing': self.testing,
                'net_reward': net_reward,
                'alpha': self.agent.learn_rate,
                'epsilon': self.agent.epsilon,
                'gamma': self.agent.gamma
            })

    def run(self, episodes=5000, n_test=0):
        state = self.env.reset()

        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, 4])
            net_reward = 0.0

            if (e % 100) == 0:
                display = True
            else:
                display = False

            for time_t in range(5000):
                if display:
                    self.env.render()
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state

                net_reward += reward

                if done:
                    print("episode: {}/{}, , e = {}, score = {}".format(e, episodes, self.agent.epsilon, time_t))
                    break
            if e > 32 and self.agent.epsilon > 0.0:
                self.agent.learn(batch_size=32)
            elif self.agent.epsilon == 0.0 and not self.testing:
                self.testing = True
            self.log_trial(e, net_reward)
        if self.log_metrics:
            self.log_file.close()

env = gym.make("CartPole-v1")
s = Simulator(env, display=True, log_metrics=True)
s.run(episodes=2000)

# def run(n_test = 0):
#     env = gym.make("CartPole-v1")
#     state = env.reset()
#     agent = Agent(env.observation_space.shape[0], env.action_space.n)
#
#     episodes = 5000
#
#     for e in range(episodes):
#         state = env.reset()
#         state = np.reshape(state, [1, 4])
#
#         if (e % 100) == 0:
#             display = True
#         else:
#             display = False
#         for time_t in range(500):
#             if display:
#                 env.render()
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             next_state = np.reshape(next_state, [1, 4])
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state
#
#             if done:
#                 print("episode: {}/{}, , e = {}, score = {}".format(e, episodes, agent.epsilon, time_t))
#                 break
#         if e >= 32 and agent.epsilon > 0.0:
#             agent.learn(batch_size=32)
