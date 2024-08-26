import os
import psutil
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import gym
import time

from aurora_lib.utils import *


class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_):
        input_ = torch.FloatTensor(input_)
        output = self.relu(self.fc1(input_))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)
        if not FLAG_CONTINUOUS_ACTION:
            output = self.softmax(output)

        return output


class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input_):
        input_ = torch.FloatTensor(input_)
        output = self.relu(self.fc1(input_))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)

        return output


def calc_gae(rewards):
    returns = []
    for episode_rewards in reversed(rewards):
        discounted_return = 0.0
        # Caution: Episodes might have different lengths if stopped earlier
        for reward in reversed(episode_rewards):
            discounted_return = reward + discounted_return * DISCOUNT
            returns.insert(0, discounted_return)

    returns = torch.FloatTensor(returns)
    return returns


class PPO:
    def __init__(self, env, verbose=True):
        self.env = env
        self.verbose = verbose

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        self.actor = ActorNetwork(self.state_size, HIDDEN_SIZE, self.action_size)
        self.critic = CriticNetwork(self.state_size, HIDDEN_SIZE, 1)

        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR)

        self.cov = torch.diag(torch.ones(self.action_size, ) * 0.5)

        self.skip_update = False

        self.num_same_parameter_actor = 0
        self.num_same_parameter_critic = 0
        self.parameter_actor = None
        self.parameter_critic = None

        # self.inf_f = open("inference_ppo.txt", "w")
        self.update_f = open("update_ppo.txt", "w")

    # skip update for the policy and critic network, i.e., policy evaluation/serving stage
    def disable_update(self):
        self.skip_update = True

    # enable update for the policy and critic network, i.e., policy training stage
    def enable_update(self):
        self.skip_update = False

    # set the environment to interact with
    def set_env(self, env):
        self.env = env

    # get the action based on the state
    def calc_action(self, state):
        if FLAG_CONTINUOUS_ACTION:
            mean = self.actor(state)
            dist = torch.distributions.MultivariateNormal(mean, self.cov)
        else:
            action_probs = self.actor(state)
            dist = torch.distributions.Categorical(action_probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    # model training
    def train(self, total_timesteps=MAX_NUM_TIMESTEPS, callback=None):
        # create the log directory if needed
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        # for RL learning curve plots
        iteration_rewards = []
        smoothed_rewards = []

        pid = os.getpid()
        python_process = psutil.Process(pid)
        steps_trained = 0
        n_validation_called = 0
        for iteration in range(TOTAL_ITERATIONS):
            if steps_trained >= total_timesteps:
                break
            print('********** Iteration %i ************' % iteration)
            steps_per_iteration = 0

            # get resource consumption profiles
            memory_usage = python_process.memory_info()[0] / 2. ** 20  # memory use in MB
            cpu_util = python_process.cpu_percent(interval=None)/psutil.cpu_count()
            print('RL Agent Memory Usage:', memory_usage, 'MB', '| CPU Util:', cpu_util)

            states = []
            actions = []
            rewards = []
            log_probs = []
            for episode in range(EPISODES_PER_ITERATION):
                state = self.env.reset()
                episode_rewards = []

                # roll out a trajectory (or so called episode)
                done = False
                steps_per_ep = 0
                while not done:
                    # t_start = int(time.time() * 1000)
                    action, log_prob = self.calc_action(state)
                    # print("Time used for inference: {:.2f}ms".format(int(time.time() * 1000) - t_start))
                    # self.inf_f.write(str(int(time.time() * 1000) - t_start) + '\n')
                    # self.inf_f.flush()

                    # clip the action to avoid out of bound error
                    if isinstance(self.env.action_space, gym.spaces.Box):
                        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

                    next_state, reward, done, _ = self.env.step(action)
                    steps_trained += 1
                    steps_per_ep += 1
                    steps_per_iteration += 1

                    states.append(state)
                    episode_rewards.append(reward)
                    log_probs.append(log_prob)
                    if FLAG_CONTINUOUS_ACTION:
                        actions.append(action)
                    else:
                        actions.append(action.item())

                    state = next_state

                    # evaluate the current policy at each checkpoint
                    if callback != None:
                        callback.on_step()

                    # check if reaching the max steps limit
                    if steps_per_ep >= MAX_TIMESTEPS_PER_EPISODE:
                        break

                # end of one episode
                print('End of an episode:', steps_per_ep, 'steps executed')
                rewards.append(episode_rewards)

            # end of one iteration
            iteration_rewards.append(np.mean([np.sum(episode_rewards) for episode_rewards in rewards]))
            smoothed_rewards.append(np.mean(iteration_rewards[-10:]))

            # states = torch.FloatTensor(states)
            # states = states.reshape((-1,) + self.observation_space.shape)
            states = torch.FloatTensor(np.array(states))
            if FLAG_CONTINUOUS_ACTION:
                actions = torch.FloatTensor(actions)
            else:
                actions = torch.IntTensor(actions)
            log_probs = torch.FloatTensor(log_probs)

            average_rewards = np.mean([np.sum(episode_rewards) for episode_rewards in rewards])
            print('Iteration:', iteration, '('+str(steps_per_iteration)+' steps) - Average rewards across episodes:', np.round(average_rewards, decimals=3),
                  '| Moving average:', np.round(np.mean(iteration_rewards[-10:]), decimals=3), '\n')

            if SAVE_TO_FILE:
                all_rewards = [reward for reward_ep in rewards for reward in reward_ep]
                self.save_trajectories(iteration, states, actions, all_rewards)
                print('Trajectory saved!')

            if self.skip_update:
                continue

            t_start = time.time()
            returns = calc_gae(rewards)

            values = self.critic(states).squeeze()
            advantage = returns - values.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for epoch in range(SGD_EPOCHS):
                batch_size = states.size(0)  # whole batch size = num of steps
                # use mini-batch instead of the whole batch
                for mini_batch in range(batch_size // MINI_BATCH_SIZE):
                    ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)

                    values = self.critic(states[ids]).squeeze()
                    if FLAG_CONTINUOUS_ACTION:
                        mean = self.actor(states[ids])
                        dist = torch.distributions.MultivariateNormal(mean, self.cov)
                    else:
                        action_probs = self.actor(states[ids])
                        dist = torch.distributions.Categorical(action_probs)

                    log_probs_new = dist.log_prob(actions[ids])
                    entropy = dist.entropy().mean()

                    ratios = (log_probs_new - log_probs[ids]).exp()

                    surrogate1 = ratios * advantage[ids]
                    surrogate2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * advantage[ids]
                    actor_loss = - torch.min(surrogate1, surrogate2).mean()
                    critic_loss = (returns[ids] - values).pow(2).mean()

                    loss = actor_loss + CRITIC_LOSS_DISCOUNT * critic_loss - ENTROPY_COEFFICIENT * entropy
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # check if the model parameters are still being updated or converged
                if self.parameter_actor is None:
                    self.parameter_actor = []
                    for parameter in self.actor.parameters():
                        self.parameter_actor.append(parameter.clone())
                else:
                    # compare the model parameters
                    is_equal = True
                    for idx, parameter in enumerate(list(self.actor.parameters())):
                        if not torch.equal(parameter, self.parameter_actor[idx]):
                            is_equal = False
                            break
                    if is_equal:
                        self.num_same_parameter_actor += 1
                    else:
                        self.num_same_parameter_actor = 0
                        self.parameter_actor = []
                        for parameter in self.actor.parameters():
                            self.parameter_actor.append(parameter.clone())
                if self.parameter_critic is None:
                    self.parameter_critic = []
                    for parameter in self.critic.parameters():
                        self.parameter_critic.append(parameter.clone())
                else:
                    # compare the model parameters one by one
                    is_equal = True
                    for idx, parameter in enumerate(list(self.critic.parameters())):
                        if not torch.equal(parameter, self.parameter_critic[idx]):
                            is_equal = False
                            break
                    if is_equal:
                        self.num_same_parameter_critic += 1
                    else:
                        self.num_same_parameter_critic = 0
                        # self.parameter_critic = list(self.critic.parameters())
                        self.parameter_critic = []
                        for parameter in self.critic.parameters():
                            self.parameter_critic.append(parameter.clone())

            print("Time used for model update: {:.2f}s".format(time.time() - t_start))
            self.update_f.write(str(time.time() - t_start) + '\n')
            self.update_f.flush()
            if self.num_same_parameter_critic > MAX_SAME_ITERATIONS and\
                    self.num_same_parameter_actor > MAX_SAME_ITERATIONS:
                print('Model parameters are not updating! Turning to policy-serving stage...')
                self.skip_update = True

        # write iteration rewards to file
        if SAVE_TO_FILE:
            file = open(LOG_DIR + "ppo_smoothed_rewards.txt", "w")
            for reward in smoothed_rewards:
                file.write(str(reward) + "\n")
            file.close()
            file = open(LOG_DIR + "ppo_iteration_rewards.txt", "w")
            for reward in iteration_rewards:
                file.write(str(reward) + "\n")
            file.close()

    # load all model parameters from a saved checkpoint
    def load_checkpoint(self, checkpoint_file_path):
        if os.path.isfile(checkpoint_file_path):
            if self.verbose:
                print('Loading checkpoint...')
            checkpoint = torch.load(checkpoint_file_path)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.verbose:
                print('Checkpoint successfully loaded!')
        else:
            raise OSError('Checkpoint not found!', checkpoint_file_path)

    # save all model parameters to a checkpoint
    def save_checkpoint(self, episode_num):
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        checkpoint_name = CHECKPOINT_DIR + 'ppo-ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(checkpoint, checkpoint_name)

    def save_model_to_path(self, model_path):
        print("saving the model to ", model_path)
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(checkpoint, model_path)

    # record trajectories
    def save_trajectories(self, episode_num, states, actions, rewards):
        file = open(LOG_DIR + "ppo_trajectories.csv", "a")
        count = 0
        for state in states:
            file.write(str(episode_num) + ',' + ','.join([str(item.item()) for item in state]) + ',' + str(actions[count].item()) + ',' + str(rewards[count]) + "\n")
            count += 1
        file.close()

    # predict the action given the observation
    def predict(self, observation):
        # observation = np.array(observation)
        # observation = observation.reshape((-1,) + self.env.observation_space.shape)
        action, _ = self.calc_action(observation)

        # clip the action to avoid out of bound error
        if isinstance(self.env.action_space, gym.spaces.Box):
            clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        return clipped_action
