from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

num_steps_per_rollout = 5
num_updates = 1000
reset_every = 200
val_every = 1000

def log(writer, iteration, name, value, print_every=10, log_every=10):
    # A simple function to let you log progress into console and tensorboard.
    if np.mod(iteration, print_every) == 0:
        if name is not None:
            print('{:8d}{:>30s}: {:0.3f}'.format(iteration, name, value))
        else:
            print('')
    if name is not None and np.mod(iteration, log_every) == 0:
        writer.add_scalar(name, value, iteration)

# Starting off from states in envs, rolls out num_steps for each environment
# envs using the policy in model. Returns rollouts in the form of states,
# actions, rewards and new states. Also returns the state the environments end
# up in after num_steps time steps.
def collect_rollouts(model, envs, states, num_steps, device):
    rollouts = []

    for i in range(num_steps):
        for j in range(len(envs)):
            env = envs[j]
            state = torch.from_numpy(states[j]).float().unsqueeze(0).to(device)
            act = model.act(state).squeeze(0)
            new_state, rewards, _ , _ = env.step(act)
            new_state = torch.from_numpy(new_state).float().unsqueeze(0).to(device)
            rollouts.append([state, act, rewards, new_state])
            states[j] = new_state

    return rollouts, states

# Using the rollouts returned by collect_rollouts function, updates the actor
# and critic models. You will need to:
# 1a. Compute targets for the critic using the current critic in model.
# 1b. Compute loss for the critic, and optimize it.
# 2a. Compute returns, or estimate for returns, or advantages for updating the actor.
# 2b. Set up the appropriate loss function for actor, and optimize it.
# Function can return actor and critic loss, for plotting.
def update_model(model, gamma, optim, rollouts, device, iteration, writer):
    actor_loss, critic_loss = 0., 0.
    for i in range(len(rollouts)):
        s, a, r, ns = rollouts[i]
        actor, critic = model.forward(s)
        n_actor, n_critic = model.forward(ns)
        target = r + gamma * n_critic
        loss_c = F.mse_loss(critic, target)

        err = r + gamma * n_critic - critic
        actor_dist = Categorical(logits=actor)
        loss_a = - actor_dist.log_prob(a) * err
        loss = loss_c + loss_a

        optim.zero_grad()
        loss.backward()
        optim.step()

        actor_loss += loss_a.view([])
        critic_loss += loss_c.view([])

    return actor_loss, critic_loss


def train_model_ac(model, envs, gamma, device, logdir, val_fn):
    model.to(device)
    train_writer = SummaryWriter(logdir / 'train')
    val_writer = SummaryWriter(logdir / 'val')

    # You may want to setup an optimizer, loss functions for training.
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Resetting all environments to initialize the state.
    num_steps, total_samples = 0, 0
    states = [e.reset() for e in envs]
    states = np.array(states)

    for updates_i in range(num_updates):

        # Put model in training mode.
        model.train()

        # Collect rollouts using the policy.
        rollouts, states = collect_rollouts(model, envs, states, num_steps_per_rollout, device)
        num_steps += num_steps_per_rollout
        total_samples += num_steps_per_rollout*len(envs)


        # Use rollouts to update the policy and take gradient steps.
        actor_loss, critic_loss = update_model(model, gamma, optim, rollouts,
                                               device, updates_i, train_writer)
        log(train_writer, updates_i, 'train-samples', total_samples, 100, 10)
        log(train_writer, updates_i, 'train-actor_loss', actor_loss, 100, 10)
        log(train_writer, updates_i, 'train-critic_loss', critic_loss, 100, 10)
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
            model.eval()
            mean_reward = val_fn(model, device)
            log(val_writer, total_samples, 'val-mean_reward', mean_reward, 1, 1)
            log(val_writer, total_samples, None, None, 1, 1)
            model.train()
