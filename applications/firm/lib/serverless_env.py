import pandas as pd
import random
from util import *


class SimEnvironment:
    # initial states
    initial_arrival_rate = 1
    initial_cpu_shares = 2048
    initial_cpu_shares_others = 0
    memory = 256
    cpu_dictionary = {0:128, 1:256, 2:512, 3:1024, 4:1536, 5:2048}
    # tabular data
    table = {}

    last_action = {
        'vertical': 0,  # last vertical action
        'horizontal': 0,  # last horizontal action
        'last_vertical': 0,  # the vertical action before last action
        'last_horizontal': 0  # the horizontal action before last action
    }
    last_reward = 0

    def __init__(self, input_path, pool=None):
        # add one example function
        self.original_cpu_util = 0.6
        self.function = 'example_function'
        self.arrival_rate = self.initial_arrival_rate
        self.cpu_shares_others = self.initial_cpu_shares_others
        self.cpu_shares_per_container = self.initial_cpu_shares
        self.num_containers = 1
        self.current_state = None
        self.pool = pool  # a list of csv files for a pool of applications
        if pool:
            self.data_path = input_path + pool[0]
            self.load_data()
        else:
            self.data_path = input_path
            self.load_data()

    # load all data from traces
    def load_data(self):
        df = pd.read_csv(self.data_path)
        for index, row in df.iterrows():
            tabular_item = {
                'avg_cpu_util': row['avg_cpu_util'],
                'slo_preservation': row['slo_preservation'],
                'total_cpu_shares': row['total_cpu_shares'],
                'cpu_shares_others': row['cpu_shares_others'],
                'num_containers':row['num_containers'],
                'arrival_rate':row['arrival_rate'],
                'latency': row['latency']
            }
            key = (row['cpu'], row['memory'])
            self.table[key] = tabular_item
        self.max_value = df.max().to_dict()
        # for the purpose of sanity check
        # for cpu in self.cpu_dictionary.values():
        #     print('CPU =', cpu)
        #     state = self.get_rl_states(cpu, 256)
        #     print('States:', state)
        #     reward = convert_state_action_to_reward(state, None, self.last_action, None)
        #     print('Reward =', reward, 'CPU util =', state[0], 'SLO preservation =', state[1])
        # exit()
        random_key,random_value = random.choice(list(self.table.items()))
        if self.pool:
            self.function = 'pool'
        else:
            self.function = self.data_path.split('/')[-1][:-11]
        init_cpu = random_key[0]
        init_cpu_util = random_value['avg_cpu_util']
        self.original_cpu_util = init_cpu_util
        self.cpu_shares_per_container = init_cpu

    # get the function name
    def get_function_name(self):
        return self.function

    # return the states
    # [cpu util percentage, slo preservation percentage, cpu.shares, cpu.shares (others), # of containers, arrival rate]
    def get_rl_states(self, cpu, memory):
        # if num_containers > MAX_NUM_CONTAINERS:
        #     return None
        value = self.table[(cpu, memory)]
        max_cpu_shares_others = self.max_value['cpu_shares_others']
        max_total_cpu_shares = self.max_value['total_cpu_shares']
        max_num_containers = self.max_value['num_containers']
        max_arrival_rate  = self.max_value['arrival_rate']
        max_latency = self.max_value['latency']
        return [value['avg_cpu_util'], value['slo_preservation'], value['total_cpu_shares']/max_total_cpu_shares,
                value['cpu_shares_others'], value['num_containers']/max_num_containers, value['arrival_rate']/max_arrival_rate, value['latency']/max_latency]

    # overprovision to num of concurrent containers + 2
    def overprovision(self, function_name):
        # horizontal scaling
        scale_action = {
            'vertical': 0,
            'horizontal': int(self.initial_arrival_rate) + 2
        }
        self.num_containers += int(self.initial_arrival_rate) + 2
        states, _, _ = self.step(function_name, scale_action)
        print('Overprovisioning:',
              '[ Avg CPU utilization: {:.7f} Avg SLO preservation: {:.7f}'.format(states[0], states[1]),
              'Num of containers:', states[4],
              'CPU shares:', states[2], 'CPU shares (others):', states[3],
              'Arrival rate:', states[5], ']')

    # reset the environment by re-initializing all states and do the overprovisioning
    def reset(self, function_name):
        if function_name != self.function:
            return KeyError

        self.arrival_rate = self.initial_arrival_rate
        self.cpu_shares_others = self.initial_cpu_shares_others
        self.num_containers = 1
        # randomly set the cpu shares for all other containers on the same server
        cpu_shares_other = 0  # single-tenant
        self.cpu_shares_others = cpu_shares_other

        # randomly set the arrival rate
        arrival_rate = 1
        self.arrival_rate = arrival_rate

        # overprovision resources to the function initially
        # self.overprovision(function_name)
        self.cpu_shares_per_container = self.initial_cpu_shares
        scale_action = {
            'vertical': random.randint(0, 5),
            'horizontal': 0
        }
        # self.num_containers += 2

        states, _, _ = self.step(function_name, scale_action)

        self.current_state = self.get_rl_states(self.cpu_shares_per_container, self.memory)

        return self.current_state

    # step function to update the environment given the input actions
    # action: +/- cpu.shares for all function containers; +/- number of function containers with the same cpu.shares
    # states: [cpu util percentage, slo preservation percentage, cpu.shares, cpu.shares (others), # of containers,
    #          arrival rate]
    def step(self, function_name, action):
        if function_name != self.function:
            raise KeyError
        curr_state = self.get_rl_states(self.cpu_shares_per_container, self.memory)
        self.cpu_shares_per_container = self.cpu_dictionary[action['vertical']]
        state = self.get_rl_states(self.cpu_shares_per_container, self.memory)
        # calculate the reward
        reward = convert_state_action_to_reward(state, action, self.last_action, self.arrival_rate)
        self.last_reward = reward
        # self.last_action = action
        self.last_action['last_vertical'] = self.last_action['vertical']
        self.last_action['last_horizontal'] = self.last_action['horizontal']
        self.last_action['vertical'] = action['vertical']
        self.last_action['horizontal'] = action['horizontal']
        # check if done
        done = False
        self.current_state = state

        return state, reward, done

    def reset_arrival_rate(self, function_name, arrival_rate):
        if function_name != self.function:
            return KeyError
        self.arrival_rate = arrival_rate
        state = self.get_rl_states(self.num_containers, self.arrival_rate)

        return state

    # print function state information
    def print_info(self):
        print('Function name:', self.function)
        print('Average CPU Util:', self.current_state[0])
        print('SLO Preservation:', self.current_state[1])
        print('Total CPU Shares (normalized):', self.current_state[2])
        print('Total CPU Shares for Other Containers (normalized):', self.current_state[3])
        print('Number of Containers:', self.current_state[4] * 20)
        print('Arrival Rate (rps):', self.current_state[5] * 10)


if __name__ == '__main__':
    print('Testing simulated environment...')
    env = SimEnvironment()
    env.load_data()
