import gym
import numpy as np

from Agent import Agent

env = gym.make("CartPole-v1")
state = env.reset()
agent = Agent(env.observation_space.shape[0], env.action_space.n)

episodes = 5000

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 4])

    if (e % 100) == 0:
        display = True
    else:
        display = False
    for time_t in range(500):
        if display:
            env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print("episode: {}/{}, , e = {}, score = {}".format(e, episodes, agent.epsilon, time_t))
            break
    if e >= 32 and agent.epsilon > 0.0:
        agent.learn(batch_size=32)
