import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

num_steps_per_rollout = 5
num_updates = 5000
reset_every = 200
val_every = 5000

replay_buffer_size = 1000000
q_target_update_every = 50
q_batch_size = 256
q_num_steps = 1

def log(writer, iteration, name, value, print_every=10, log_every=10):
    # A simple function to let you log progress into console and tensorboard.
    if np.mod(iteration, print_every) == 0:
        if name is not None:
            print('{:8d}{:>30s}: {:0.3f}'.format(iteration, name, value))
        else:
            print('')
    if name is not None and np.mod(iteration, log_every) == 0:
        writer.add_scalar(name, value, iteration)

# You may want to make a replay buffer class that a) stores rollouts as they
# come along, overwriting older rollouts as needed, and b) allows random
# sampling of transition quadruples for training of the Q-networks.
class ReplayBuffer(object):
    def __init__(self, size, state_dim, action_dim):
        self.size = size
        self.states = torch.zeros(size, state_dim)
        self.actions = torch.zeros(size, action_dim)
        self.rewards = torch.zeros(size)
        self.new_states = torch.zeros(size, state_dim)
        self.current = 0        # count rollouts for overwritting in time
        self.full = 0

    def insert(self, rollouts):
        # rollouts = [states, actions, rewards, new_states]
        self.states[self.current] = rollouts[0]
        self.actions[self.current] = rollouts[1]
        self.rewards[self.current] = rollouts[2]
        self.new_states[self.current] = rollouts[3]
        if self.current + 1 == self.size:
            self.full = 1
        self.current = (self.current + 1) % self.size


    def sample_batch(self, batch_size):
        random_indices = np.random.choice(max(self.current, self.size*self.full), batch_size, replace = False)
        s, a, r, ns = self.states[random_indices], self.actions[random_indices], self.rewards[random_indices], self.new_states[random_indices]
        return s, a, r, ns

# Starting off from states in envs, rolls out num_steps_per_rollout for each
# environment envs using the policy in model. Returns rollouts in the form of
# states, actions, rewards and new states. Also returns the state the
# environments end up in after num_steps_per_rollout time steps.
def collect_rollouts(models, envs, states, num_steps_per_rollout, epsilon, device):
    rollouts = []
    # TODO
    model = models[0]
    for i in range(num_steps_per_rollout):
        for j in range(len(envs)):
            env = envs[j]
            state = torch.from_numpy(states[j]).float().unsqueeze(0).to(device)
            act = model.act(state, epsilon).squeeze(0)
            new_state, rewards, _ , _ = env.step(act)
            new_state = torch.from_numpy(new_state).float().unsqueeze(0).to(device)
            rollouts.append([state, act, rewards, new_state])
            states[j] = new_state

    return rollouts, states

# Function to train the Q function. Samples q_num_steps batches of size
# q_batch_size from the replay buffer, runs them through the target network to
# obtain target values for the model to regress to. Takes optimization steps to
# do so. Returns the bellman_error for plotting.
def update_model(replay_buffer, models, targets, optim, gamma, action_dim,
                 q_batch_size, q_num_steps):
    total_bellman_error = 0.
    for i in range(q_num_steps):
        sample = replay_buffer.sample_batch(q_batch_size)
        for j in range(q_batch_size):
            s, a, r, ns = sample[0][j], sample[1][j], sample[2][j], sample[3][j]

            Q_current = models[0].forward(s).gather(0, a.type(torch.int64))
            Q_next = targets[0].forward(ns).max().view(1)
            Q_target = r + gamma * Q_next

            loss = F.mse_loss(Q_current, Q_target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_bellman_error += loss

    return total_bellman_error

def train_model_dqn(models, targets, state_dim, action_dim, envs, gamma, device, logdir, val_fn):
    train_writer = SummaryWriter(logdir / 'train')
    val_writer = SummaryWriter(logdir / 'val')

    # Set model into training mode
    [m.train() for m in models]

    # You may want to setup an optimizer, loss functions for training.
    optim = torch.optim.Adam(models[0].parameters(), lr=1e-4)

    # Set up the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, state_dim, 1)

    # Resetting all environments to initialize the state.
    num_steps, total_samples = 0, 0
    states = [e.reset() for e in envs]
    states = np.array(states)
    epsilon = 1

    for updates_i in range(num_updates):
        # Come up with a schedule for epsilon
        if total_samples % 100 == 0:
            epsilon = max(epsilon * 0.995, 0.01)

        # Put model in training mode.
        [m.train() for m in models]

        if np.mod(updates_i, q_target_update_every) == 0:
            # If you are using a target network, every few updates you may want
            # to copy over the model to the target network.
            # TODO
            for i in range(len(models)):
                targets[i].load_state_dict(models[i].state_dict())
            # targets = models.copy()

        # Collect rollouts using the policy.
        rollouts, states = collect_rollouts(models, envs, states, num_steps_per_rollout, epsilon, device)
        num_steps += num_steps_per_rollout
        total_samples += num_steps_per_rollout*len(envs)

        # Push rollouts into the replay buffer.
        for i in rollouts:
            replay_buffer.insert(i)

        # Use replay buffer to update the policy and take gradient steps.
        if total_samples > q_batch_size:
            bellman_error = update_model(replay_buffer, models, targets, optim,
                                     gamma, action_dim, q_batch_size,
                                         q_num_steps)
            log(train_writer, updates_i, 'train-samples', total_samples, 100, 10)
            log(train_writer, updates_i, 'train-bellman-error', bellman_error, 100, 10)
            log(train_writer, updates_i, 'train-epsilon', epsilon, 100, 10)
            log(train_writer, updates_i, None, None, 100, 10)


        # We are solving a continuing MDP which never returns a done signal. We
        # are going to manully reset the environment every few time steps. To
        # track progress on the training envirnments you can maintain the
        # returns on the training environments, and log or print it out when
        # you reset the environments.
        if num_steps >= reset_every:
            states = [e.reset() for e in envs]
            states = np.array(states)
            num_steps = 0

        # Every once in a while run the policy on the environment in the
        # validation set. We will use this to plot the learning curve as a
        # function of the number of samples.
        cross_boundary = total_samples // val_every > \
            (total_samples - len(envs)*num_steps_per_rollout) // val_every
        if cross_boundary:
            models[0].eval()
            mean_reward = val_fn(models[0], device)
            log(val_writer, total_samples, 'val-mean_reward', mean_reward, 1, 1)
            log(val_writer, total_samples, None, None, 1, 1)
            models[0].train()
