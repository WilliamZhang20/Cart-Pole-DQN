# Cart Pole DQN

In classic Q-learning, a Q-table was used to store the expected utility values for each state-action pair.
But for more complex examples such as a cart pole control environment, it is more optimal to use a neural network to map states, and actions to a reward. 

This is called a Deep Q Network (DQN), as described in [this](https://arxiv.org/abs/1312.5602) very famous paper.

In this repository, I have trained a DQN to control a cart pole and keep it vertical.

The file `agent.py` contains methods to train and execute actions for the cart pole controller agent, and the file `train_cartpole.py` runs training cycles and captures a video of the controller in action!

So far, training takes an incredible number of time, between 30 minutes and an hour on a x86-64 CPU running an Intel i5.

*Remains a work in progress*.