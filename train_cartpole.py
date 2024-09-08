from agent import Agent
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np

def train_agent():

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n # only 2 actions left or right!

    n_episodes = 1000 # maybe 450 - 500
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
        print("Episode:", e)

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
    agent.save('trained_agent.pkl')
    return agent

def make_video(agent):
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = RecordVideo(env, video_folder="cartpole-video", name_prefix="eval",
                  episode_trigger=lambda x: True)

    rewards = 0
    steps = 0
    done = False
    state, _ = env.reset() # split out the tuple
    state = np.array([state])
    agent.exp_prob = 0 # block out all explorations

    while not done:
        action = agent.compute_action(state)
        state, reward, done, _, _ = env.step(action)
        state = np.array([state])
        steps += 1
        rewards += reward

    print(rewards)
    env.close()

option = input()
if option == "1":
    agent = train_agent()
else:
    agent = Agent.load('trained_agent.pkl')
make_video(agent)