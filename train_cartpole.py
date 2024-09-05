from agent import Agent
import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n # 

n_episodes = 400
max_iteration_ep = 500

# Agent defined
agent = Agent(state_size, action_size)
total_steps = 0

for e in range(n_episodes):
    current_state, _ = env.reset() # env.reset() returns a tuple, we extract
    # print("Type of current_state:", type(current_state))
    # print("Current state shape:", np.shape(current_state))
    current_state = np.array(current_state, dtype=np.float32)
    current_state = np.reshape(current_state, [1, state_size])

    for step in range(max_iteration_ep):
        total_steps = total_steps + 1
        # Agent computes actions in training setting
        action = agent.compute_action(current_state)
        # Run the action
        next_state, reward, done, _, _ = env.step(action) # length of step tuple result is 5

        next_state = np.array([next_state])

        agent.store_episode(current_state, action, reward, next_state, done)

        if done:
            agent.update_exp_prob()
            break

        current_state = next_state
    
    if total_steps >= agent.batch_size:
        agent.train()

print("Done training")

def make_video():
    env = gym.make('CartPole-v1')
    env.record('videos')

    rewards = 0
    steps = 0
    done = False
    state = env.reset()
    state = np.array([state])

    while not done:
        action = agent.compute_action(state)
        state, reward, done, _ = env.step(action)
        state = np.array([state])
        steps += 1
        rewards += reward

    print(rewards)
    env.close()

make_video()