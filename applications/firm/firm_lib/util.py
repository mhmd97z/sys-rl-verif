import numpy as np
import os


NUM_STATES = 7
NUM_GLOBAL_STATES = 4
NUM_GLOBAL_STATES_WITH_VARIANCE = NUM_GLOBAL_STATES * 2
NUM_MEAN_FIELD_STATES = NUM_STATES
NUM_MEAN_FIELD_STATES_WITH_ACTIONS = NUM_STATES + 2  # 2 for horizontal and vertical actions
NUM_ACTIONS = 6
NUM_TOTAL_ACTIONS = 10
VERTICAL_SCALING_STEP = 128
HORIZONTAL_SCALING_STEP = 1
MAX_NUM_CONTAINERS = 10.0
MAX_CPU_SHARES = 2048.0
cpu_dictonary = {0:128, 1:256, 2:512, 3:1024, 4:1536, 5:2048}
ILLEGAL_PENALTY = -1

RESULTS_PATH = "./results-testing"
POOL_ID = int(os.environ.get('POOL_ID_ENV', -1))
if POOL_ID != -1:
    RESULTS_PATH = "./results-pool-" + str(POOL_ID)

"""
PPO
"""
PLOT_FIG = False
SAVE_FIG = False
SAVE_TO_FILE = False
FLAG_BERT_TINY = True

META_TRAIN_ITERATIONS = 200
TOTAL_ITERATIONS = 4
TOTAL_TEST_ITERATIONS = 5
EPISODES_PER_ITERATION = 4
EPISODE_LENGTH = 50
WARMUP_STEPS = 2500

# hyperparameters
DISCOUNT = 0.99
HIDDEN_SIZE = 64
LR = 3e-4  # 5e-3 5e-6
BERT_LR = 1e-5
FINE_TUNE_LR = 1e-4
SGD_EPOCHS = 5
MINI_BATCH_SIZE = 5
CLIP = 0.2
ENTROPY_COEFFICIENT = 0.01  # 0.001
CRITIC_LOSS_DISCOUNT = 0.05  # 0.03

FLAG_CONTINUOUS_ACTION = False

MAX_SAME_ITERATIONS = 2 * SGD_EPOCHS

"""
Meta-learning
"""
BUFFER_UPDATE_MODE = 'best'
BUFFER_SIZE = 8

NUM_FEATURES_PER_SHOT = 9  # 7 (obs) + 1 (action) + 1 (reward) for WA
RNN_HIDDEN_SIZE = EPISODE_LENGTH  # equal to the max steps of each episode
RNN_NUM_LAYERS = 2
EMBEDDING_DIM = 64

# print current state
def print_state(state_list):
    print('Avg CPU util: {:.7f} Avg SLO preservation: {:.7f}'.format(state_list[0], state_list[1]),
          'Num of containers:', state_list[4],
          'CPU shares:', state_list[2], 'CPU shares (others):', state_list[3],
          'Arrival rate:', state_list[4])


# print current action
def print_action(action_dict):
    if action_dict['vertical'] > 0:
        print('Action: Scaling-up by', VERTICAL_SCALING_STEP, 'cpu.shares')
    elif action_dict['vertical'] < 0:
        print('Action: Scaling-down by', VERTICAL_SCALING_STEP, 'cpu.shares')
    elif action_dict['horizontal'] > 0:
        print('Action: Scaling-out by', HORIZONTAL_SCALING_STEP, 'container')
    elif action_dict['horizontal'] < 0:
        print('Action: Scaling-in by', HORIZONTAL_SCALING_STEP, 'container')
    else:
        print('No action to perform')


# print (state, action, reward) for the current step
def print_step_info(step, state_list, action_dict, reward):
    state = 'State: [Avg CPU utilization: {:.7f} Avg SLO preservation: {:.7f}'.format(state_list[0], state_list[1]) +\
            ' Num of containers: ' + str(state_list[4]) +\
            ' CPU shares: ' + str(state_list[2]) + ' CPU shares (others): ' + str(state_list[3]) +\
            ' Arrival rate: ' + str(state_list[5]) + ']'
    action = 'Action: N/A'
    action = 'Action: Scaling to ' + str(cpu_dictonary[action_dict['vertical']]) + ' cpu.shares'
    print('Step #' + str(step), '|', state, '|', action, '| Reward:', reward)

def print_step_info_with_function_name(step, state_list, action_dict, reward, function_name):
    state = 'State: [Avg CPU utilization: {:.7f} Avg SLO preservation: {:.7f}'.format(state_list[0], state_list[1]) +\
            ' Num of containers: ' + str(state_list[4]) +\
            ' CPU shares: ' + str(state_list[2]) + ' CPU shares (others): ' + str(state_list[3]) +\
            ' Arrival rate: ' + str(state_list[5]) + ']'
    action = 'Action: N/A'
    if action_dict['vertical'] > 0:
        action = 'Action: Scaling-up by ' + str(VERTICAL_SCALING_STEP) + ' cpu.shares'
    elif action_dict['vertical'] < 0:
        action = 'Action: Scaling-down by ' + str(VERTICAL_SCALING_STEP) + ' cpu.shares'
    elif action_dict['horizontal'] > 0:
        action = 'Action: Scaling-out by ' + str(HORIZONTAL_SCALING_STEP) + ' container'
    elif action_dict['horizontal'] < 0:
        action = 'Action: Scaling-in by ' + str(HORIZONTAL_SCALING_STEP) + ' container'
    print('[', function_name, '] - Step #' + str(step), '|', state, '|', action, '| Reward:', reward)

# calculate the reward based on the current states (after the execution of the current action)
# + cpu utilization percentage [0, 1]
# + slo preservation [0, 1]
# - number of containers (/ arrival rate)
# - penalty
def convert_state_action_to_reward(state, action, last_action, arrival_rate):

    alpha = 0.3  # 0.3  - cpu-util
    beta = 0.7  # 0.7  - slo-preservation
    reward = alpha * state[0] + beta * state[1] #+ (1 - alpha - beta) * gamma * reward_less_containers

    if (action['vertical'] - last_action['vertical']) * (last_action['vertical'] - last_action['last_vertical']) < 0:
        reward = -1  # -ILLEGAL_PENALTY

    return reward


def convert_state_action_to_reward_overprovisioning(state, action, last_action, arrival_rate):
    reward = state[1]
    return reward


def convert_state_action_to_reward_tightpacking(state, action, last_action, arrival_rate):
    reward = state[0]
    return reward


# check the correctness of the action
def is_correct(action, num_containers, cpu_shares_per_container):

    if num_containers <= 0:
        if action['vertical'] != 0 or action['horizontal'] < 0:
            return False
    else:
        if action['vertical'] + cpu_shares_per_container < 128:
            return False

    if num_containers + action['horizontal'] > MAX_NUM_CONTAINERS:
        return False
    if action['vertical'] + cpu_shares_per_container > MAX_CPU_SHARES:
        return False

    return True


# count the number of parameters in a model
def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
