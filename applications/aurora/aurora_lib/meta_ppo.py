import os
import psutil
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import gym
import heapq
import time
from torch.nn.functional import cosine_similarity

from utils import *
from rnn import RNNEmbedding


# deal with variable-sized trajectories
# one trajectory: { 'states': states_ep, 'actions': actions_ep, 'rewards': rewards_ep }
def get_padded_trajectories(trajectories, state_dim=1, action_dim=1):
    batch_size = len(trajectories)
    if batch_size == 0:
        # where the buffer is empty
        max_traj_len = MAX_TIMESTEPS_PER_EPISODE
        batch_size = 1
        padded_states = torch.zeros(batch_size, state_dim, max_traj_len)
        padded_actions = torch.zeros(batch_size, action_dim, max_traj_len)
        padded_rewards = torch.zeros(batch_size, 1, max_traj_len)
        return padded_states, padded_actions, padded_rewards

    max_traj_len = max([len(trajectories[traj_key]['states']) for traj_key in trajectories])
    # print('Maximum episode length:', max_traj_len)
    assert max_traj_len <= MAX_TIMESTEPS_PER_EPISODE
    max_traj_len = MAX_TIMESTEPS_PER_EPISODE
    
    # pad sequences in the batch to have equal length
    padded_states = torch.zeros(batch_size, state_dim, max_traj_len)
    padded_actions = torch.zeros(batch_size, action_dim, max_traj_len)
    padded_rewards = torch.zeros(batch_size, 1, max_traj_len)
    for i, traj_key in enumerate(trajectories):
        traj = trajectories[traj_key]
        for j in range(len(traj['states'])):
            for s in range(len(traj['states'][j])):
                padded_states[i][s][j] = torch.tensor(traj['states'][j][s])
            for a in range(len(traj['actions'][j])):
                padded_actions[i][a][j] = torch.tensor(traj['actions'][j][a])
            padded_rewards[i][0][j] = torch.tensor(traj['rewards'][j])

    return padded_states, padded_actions, padded_rewards


class MetaActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, env_dim, agent, verbose=True):
        super(MetaActorNetwork, self).__init__()

        self.env_dim = env_dim
        self.agent = agent

        # rnn embedding
        num_features_per_sample = env_dim['state'] + env_dim['action'] + env_dim['reward']
        self.rnn = RNNEmbedding(1, BUFFER_SIZE, 'cc', num_channels=MAX_TIMESTEPS_PER_EPISODE, verbose=verbose)

        self.fc1 = nn.Linear(input_size + self.rnn.embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.last_embedding = None

    def forward(self, input_):
        padded_states, padded_actions, padded_rewards = get_padded_trajectories(self.agent.episode_buffer, state_dim=self.env_dim['state'], action_dim=self.env_dim['action'])

        # encode RL trajectories with the RNN and generate the embedding
        rnn_input = torch.cat((padded_states, padded_actions, padded_rewards), dim=-2)
        num_sequences = rnn_input.shape[0]
        hidden = self.rnn.init_hidden(num_sequences=num_sequences)
        rnn_output, hidden_state = self.rnn.gru(rnn_input, hidden)  # use the last hidden state to generate the embedding
        rnn_output = rnn_output[:, -1, :]
        hidden_state = torch.mean(hidden_state, dim=1, keepdim=True)
        hidden_state = hidden_state.view((1, 1, -1))
        # embedding = self.rnn.embedding_layer(torch.cat((rnn_output[:, :self.rnn.hidden_size], rnn_output[:, self.rnn.hidden_size:]), dim=-1))
        embedding = self.rnn.embedding_layer(hidden_state[0][0])
        embedding = self.rnn.relu(embedding)
        # print('Generated embedding:', type(embedding), embedding.shape)
        # if self.last_embedding != None:
        #     cos_sim = cosine_similarity(embedding.unsqueeze(0), self.last_embedding.unsqueeze(0))
        #     # print('Cosine similarity:', cos_sim.item())
        #     euclidean_dist = torch.dist(embedding, self.last_embedding)
        #     # print('Euclidean distance:', euclidean_dist.item())
        # self.last_embedding = embedding

        input_ = torch.FloatTensor(input_)
        if len(input_.shape) > 1:
            # concat input + embedding for batched inputs
            embedding = embedding.reshape(1, -1)
            embedding = embedding.repeat(input_.size(0), 1)
            input_ = torch.cat((input_, embedding), dim=1)
        else:
            # concat input + embedding
            input_ = torch.cat((input_, embedding), dim=-1)
        output = self.relu(self.fc1(input_))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)
        if not FLAG_CONTINUOUS_ACTION:
            output = self.softmax(output)

        return output


class MetaCriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MetaCriticNetwork, self).__init__()

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


class MetaPPO:
    def __init__(self, env, verbose=True):
        self.env = env
        self.verbose = verbose

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        self.env_dim = {
            'state': self.state_size,
            'action': self.action_size,
            'reward': 1
        }

        self.actor = MetaActorNetwork(self.state_size, HIDDEN_SIZE, self.action_size, self.env_dim, self, verbose=verbose)
        self.critic = MetaCriticNetwork(self.state_size, HIDDEN_SIZE, 1)

        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR)

        self.cov = torch.diag(torch.ones(self.action_size, ) * 0.5)

        self.skip_update = False

        self.num_same_parameter_actor = 0
        self.num_same_parameter_critic = 0
        self.parameter_actor = None
        self.parameter_critic = None

        # episode buffer for embedding generation
        self.episode_buffer = None

        self.config = {
            'mode': BUFFER_UPDATE_MODE,
            'buffer_size': BUFFER_SIZE
        }

        self.clear_episode_buffer()
        self.cached_embedding = None
        self.use_cached_embedding = False

        # self.inf_f = open("inference_meta.txt", "w")
        # self.update_f = open("update_meta.txt", "w")

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

    # clear episode buffer
    def clear_episode_buffer(self):
        self.episode_buffer = {}
        self.episode_buffer_rewards = []
        self.episode_buffer_index = 0
        heapq.heapify(self.episode_buffer_rewards)

    # update episode buffer
    def update_episode_buffer(self, states_ep, actions_ep, rewards_ep, steps_ep):
        assert len(states_ep) == steps_ep
        assert len(actions_ep) == steps_ep
        assert len(rewards_ep) == steps_ep

        if self.config['mode'] == 'best':
            # episode buffer contains the episodes with the highest rewards
            reward = round(np.sum(rewards_ep), 5)
            if len(self.episode_buffer) >= self.config['buffer_size']:
                # check if the new episode has a higher reward than the least episode reward in the buffer
                if reward >= self.episode_buffer_rewards[0]:
                    # remove the episode with the least reward
                    removed = heapq.heappop(self.episode_buffer_rewards)
                    del self.episode_buffer[removed]
                else:
                    # ignore the new episode since its reward is smaller than any episode in the buffer
                    return

            # add the new episode and its reward to the buffer
            heapq.heappush(self.episode_buffer_rewards, reward)
            self.episode_buffer[reward] = {
                'states': states_ep,
                'actions': actions_ep,
                'rewards': rewards_ep
            }
            # cached embedding is not valid
            self.use_cached_embedding = False
        elif self.config['mode'] == 'latest':
            # episode buffer contains the lastest episodes
            if len(self.episode_buffer) >= self.config['buffer_size']:
                # remove the oldest episode
                oldest = self.episode_buffer_rewards[self.episode_buffer_index]
                del self.episode_buffer[oldest]
                self.episode_buffer[self.episode_buffer_index] = {
                    'states': states_ep,
                    'actions': actions_ep,
                    'rewards': rewards_ep
                }
                self.episode_buffer_index = (self.episode_buffer_index + 1) % self.config['buffer_size']
            else:
                # add the new episode to the buffer
                heapq.heappush(self.episode_buffer_rewards, self.episode_buffer_index)
                self.episode_buffer[self.episode_buffer_index] = {
                    'states': states_ep,
                    'actions': actions_ep,
                    'rewards': rewards_ep
                }
                self.episode_buffer_index = (self.episode_buffer_index + 1) % self.config['buffer_size']
            # cached embedding is not valid
            self.use_cached_embedding = False
        else:
            raise NotImplementedError

    # get episode buffer
    def get_episode_buffer(self):
        return [self.episode_buffer[r] for r in reversed(self.episode_buffer_rewards)]

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
                trajectory_action = []
                trajectory_state = []
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

                    trajectory_state.append(state)
                    trajectory_action.append(action)

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

                # update the episode buffer
                self.update_episode_buffer(trajectory_state, trajectory_action, episode_rewards, steps_per_ep)

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
                # save the trajectories in this iteration to local
                all_rewards = [reward for reward_ep in rewards for reward in reward_ep]
                self.save_trajectories(iteration, states, actions, all_rewards)
                print('Trajectory saved!')

            # check if model update is skipped
            if self.skip_update:
                continue

            # update RL policy
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
                print("Time used for model update: {:.2f}s".format(time.time() - t_start))
                self.update_f.write(str(time.time() - t_start) + '\n')
                self.update_f.flush()

                # # check if the model parameters are still being updated or converged
                # if self.parameter_actor is None:
                #     self.parameter_actor = []
                #     for parameter in self.actor.parameters():
                #         self.parameter_actor.append(parameter.clone())
                # else:
                #     # compare the model parameters
                #     is_equal = True
                #     for idx, parameter in enumerate(list(self.actor.parameters())):
                #         if not torch.equal(parameter, self.parameter_actor[idx]):
                #             is_equal = False
                #             break
                #     if is_equal:
                #         self.num_same_parameter_actor += 1
                #     else:
                #         self.num_same_parameter_actor = 0
                #         self.parameter_actor = []
                #         for parameter in self.actor.parameters():
                #             self.parameter_actor.append(parameter.clone())
                # if self.parameter_critic is None:
                #     self.parameter_critic = []
                #     for parameter in self.critic.parameters():
                #         self.parameter_critic.append(parameter.clone())
                # else:
                #     # compare the model parameters one by one
                #     is_equal = True
                #     for idx, parameter in enumerate(list(self.critic.parameters())):
                #         if not torch.equal(parameter, self.parameter_critic[idx]):
                #             is_equal = False
                #             break
                #     if is_equal:
                #         self.num_same_parameter_critic += 1
                #     else:
                #         self.num_same_parameter_critic = 0
                #         # self.parameter_critic = list(self.critic.parameters())
                #         self.parameter_critic = []
                #         for parameter in self.critic.parameters():
                #             self.parameter_critic.append(parameter.clone())

            # print("Time used for model update: {:.2f}s".format(time.time() - t_start))
            # self.update_f.write(str(time.time() - t_start) + '\n')
            # self.update_f.flush()
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
            raise OSError('Checkpoint not found!')

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

if __name__ == '__main__':
    # testing of the episode buffer update
    from network_simulator.pcc.aurora.schedulers import TestScheduler
    from trace import generate_trace
    from network_simulator.pcc.aurora import aurora_environment
    dummy_trace = generate_trace((10, 10), (2, 2), (2, 2), (50, 50), (0, 0), (1, 1), (0, 0), (0, 0))
    test_scheduler = TestScheduler(dummy_trace)
    env = gym.make('AuroraEnv-v0', trace_scheduler=test_scheduler)
    model = MetaPPO(env)

    model.config['buffer_size'] = 3
    model.config['mode'] = 'latest'

    model.update_episode_buffer([1, 2, 3], [1, 1, 1], [1, 2, 3], 3)
    model.update_episode_buffer([2, 3, 4], [1, 1, 1], [2, 3, 4], 3)
    model.update_episode_buffer([3, 4, 5], [1, 1, 1], [3, 4, 5], 3)
    model.update_episode_buffer([4, 5, 6], [1, 1, 1], [4, 5, 6], 3)
    model.update_episode_buffer([5, 6, 7], [1, 1, 1], [5, 6, 7], 3)
    model.update_episode_buffer([1, 2, 3], [1, 1, 1], [1, 2, 3], 3)
    model.update_episode_buffer([2, 3, 4], [1, 1, 1], [2, 3, 4], 3)
    model.update_episode_buffer([2, 3, 4], [1, 1, 1], [2, 3, 4], 3)

    for item in model.get_episode_buffer():
        print(item)
