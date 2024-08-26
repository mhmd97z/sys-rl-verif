import csv
import logging
import multiprocessing as mp
import os
import time
import types
from typing import List, Tuple, Union
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from mpi4py.MPI import COMM_WORLD

import gym
import numpy as np
import tqdm
import torch

# from stable_baselines import PPO1
# from stable_baselines.common.callbacks import BaseCallback
# from stable_baselines.common.policies import FeedForwardPolicy

from aurora_lib.network_simulator.pcc.aurora import aurora_environment
from aurora_lib.network_simulator.pcc.aurora.schedulers import Scheduler, TestScheduler
from aurora_lib.network_simulator.constants import BITS_PER_BYTE, BYTES_PER_PACKET
from aurora_lib.trace import generate_trace, Trace, generate_traces
from aurora_lib.utils import pcc_aurora_reward, MAX_TIMESTEPS_PER_EPISODE
from aurora_lib.plot_scripts.plot_packet_log import plot
from aurora_lib.plot_scripts.plot_time_series import plot as plot_simulation_log
from aurora_lib.ppo import PPO
# from meta_ppo import MetaPPO


class SaveOnBestTrainingRewardCallback():
    """
    Callback for evaluating and saving a model based on the training reward.
    The check is done every check_freq steps.

    :param aurora: (Aurora) RL agent.
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    :param val_traces: (List[Trace])
    :param verbose: (int)
    :param steps_trained: (int)
    :param config_file: (Union[str, None])
    """
    def __init__(self, aurora, check_freq: int, log_dir: str, val_traces: List[Trace] = [],
                 verbose=0, steps_trained=0, config_file: Union[str, None] =None):
        self.aurora = aurora
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = log_dir
        # create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        self.best_mean_reward = -np.inf
        self.val_traces = val_traces
        self.verbose = verbose
        # number of times env.step() was called
        self.steps_trained = steps_trained
        # number of time the callback was called
        self.n_calls = 0  # type: int
        self.config_file = config_file

        # initialize the validation log writer
        self.val_log_writer = csv.writer(
            open(os.path.join(log_dir, 'validation_log.csv'), 'w', 1),
            delimiter=',', lineterminator='\n')
        self.val_log_writer.writerow(
            ['n_calls', 'num_timesteps', 'mean_validation_reward', 'mean_validation_pkt_level_reward', 'loss',
             'throughput', 'latency', 'sending_rate', 'tot_t_used(min)',
             'val_t_used(min)', 'train_t_used(min)'])

        os.makedirs(os.path.join(log_dir, "validation_traces"), exist_ok=True)
        for i, tr in enumerate(self.val_traces):
            tr.dump(os.path.join(log_dir, "validation_traces", "trace_{}.json".format(i)))

        self.best_val_reward = -np.inf
        self.val_times = 0

        self.t_start = time.time()
        self.prev_t = time.time()

    def on_step(self) -> bool:
        self.n_calls += 1
        self.steps_trained += 1
        if self.n_calls % self.check_freq == 0:
            print('Evaluate the current learned policy...')
            # save the model checkpoint
            model_path_to_save = os.path.join(self.save_path, "model_step_{}.ckpt".format(int(self.steps_trained)))
            self.aurora.model.save_model_to_path(model_path_to_save)
            print('Saved model checkpoint to:', model_path_to_save)

            if not self.val_traces:
                return True

            # evaluate the model on validation traces
            avg_tr_bw = []
            avg_tr_min_rtt = []
            avg_tr_loss = []
            avg_rewards = []
            avg_pkt_level_rewards = []
            avg_losses = []
            avg_tputs = []
            avg_delays = []
            avg_send_rates = []
            val_start_t = time.time()

            for idx, val_trace in enumerate(self.val_traces):
                avg_tr_bw.append(val_trace.avg_bw)
                avg_tr_min_rtt.append(val_trace.avg_bw)
                ts_list, val_rewards, loss_list, tput_list, delay_list, \
                    send_rate_list, action_list, obs_list, mi_list, pkt_level_reward = self.aurora._test(
                        val_trace, self.log_dir)
                avg_rewards.append(np.mean(np.array(val_rewards)))
                avg_losses.append(np.mean(np.array(loss_list)))
                avg_tputs.append(float(np.mean(np.array(tput_list))))
                avg_delays.append(np.mean(np.array(delay_list)))
                avg_send_rates.append(
                    float(np.mean(np.array(send_rate_list))))
                avg_pkt_level_rewards.append(pkt_level_reward)
            cur_t = time.time()
            self.val_log_writer.writerow(
                map(lambda t: "%.3f" % t,
                    [float(self.n_calls), float(self.steps_trained),
                     np.mean(np.array(avg_rewards)),
                     np.mean(np.array(avg_pkt_level_rewards)),
                     np.mean(np.array(avg_losses)),
                     np.mean(np.array(avg_tputs)),
                     np.mean(np.array(avg_delays)),
                     np.mean(np.array(avg_send_rates)),
                     (cur_t - self.t_start) / 60,
                     (cur_t - val_start_t) / 60, (val_start_t - self.prev_t) / 60]))
            self.prev_t = cur_t
            print('Done!')
        return True

class Aurora():
    cc_name = 'aurora'
    def __init__(self, seed: int, log_dir: str, timesteps_per_actorbatch: int,
                 pretrained_model_path: str = "", record_pkt_log: bool = False, verbose: bool = True, enable_meta: bool = False):
        self.seed = seed
        self.log_dir = log_dir
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.pretrained_model_path = pretrained_model_path
        self.record_pkt_log = record_pkt_log
        self.verbose = verbose

        # create a dummy environment as the default environment for RL training
        self.steps_trained = 0
        dummy_trace = generate_trace(
            (10, 10), (2, 2), (2, 2), (50, 50), (0, 0), (1, 1), (0, 0), (0, 0))
        # env = gym.make('AuroraEnv-v0', traces=[dummy_trace], train_flag=True)
        test_scheduler = TestScheduler(dummy_trace)
        env = gym.make('AuroraEnv-v0', trace_scheduler=test_scheduler)

        # init an RL agent
        if not enable_meta:
            self.model = PPO(env, verbose=self.verbose)
            if self.verbose:
                print('RL agent initialized!')
            if pretrained_model_path:
                # load the model from a checkpoint
                # TODO: update steps_trained from the last checkpoint
                self.model.load_checkpoint(pretrained_model_path)
                if self.verbose:
                    print('Model loaded from', pretrained_model_path)
        else:
            # meta-learning enabled
            self.model = MetaPPO(env, verbose=self.verbose)
            if self.verbose:
                print('Meta-RL agent initialized!')
            if pretrained_model_path:
                # load the model from a checkpoint
                # TODO: update steps_trained from the last checkpoint
                self.model.load_checkpoint(pretrained_model_path)
                if self.verbose:
                    print('Model loaded from', pretrained_model_path)

    # model training
    def train(self, config_file: str, total_timesteps: int,
              train_scheduler: Scheduler,
              validation_traces: List[Trace] = [], validation_config_file: str = None):
        # generate validation traces
        if not validation_traces or len(validation_traces) == 0:
            if not validation_config_file:
                validation_traces = generate_traces(config_file, 20, duration=30)
                print('Generated 20 traces for validation using config file:', config_file)
            else:
                validation_traces = generate_traces(validation_config_file, 20, duration=30)
                print('Generated 20 traces for validation using config file:', validation_config_file)

        # create the callback: check every n steps and save best model
        self.callback = SaveOnBestTrainingRewardCallback(
            self, check_freq=self.timesteps_per_actorbatch, log_dir=self.log_dir,
            steps_trained=self.steps_trained, val_traces=validation_traces,
            config_file=config_file)

        # create the environment
        env = gym.make('AuroraEnv-v0', trace_scheduler=train_scheduler)
        env.seed(self.seed)
        # set the environment for RL model training
        self.model.set_env(env)
        print('RL environment initialized!')

        # start training
        # input("Press Enter to continue...")
        self.model.train(total_timesteps=total_timesteps, callback=self.callback)

        self.model.save_model_to_path(self.log_dir+"model")

    # evaluate the model on test traces
    def test_on_traces(self, traces: List[Trace], save_dirs: List[str]):
        results = []
        pkt_logs = []
        for trace, save_dir in zip(traces, save_dirs):
            ts_list, reward_list, loss_list, tput_list, delay_list, \
                send_rate_list, action_list, obs_list, mi_list, pkt_log = self._test(
                    trace, save_dir)
            result = list(zip(ts_list, reward_list, send_rate_list, tput_list,
                              delay_list, loss_list, action_list, obs_list, mi_list))
            pkt_logs.append(pkt_log)
            results.append(result)
        return results, pkt_logs

    def _test(self, trace: Trace, save_dir: str, plot_flag: bool = False, saliency: bool = False):
        reward_list = []
        loss_list = []
        tput_list = []
        delay_list = []
        send_rate_list = []
        ts_list = []
        action_list = []
        mi_list = []
        obs_list = []
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            f_sim_log = open(os.path.join(save_dir, '{}_simulation_log.csv'.format(self.cc_name)), 'w', 1)
            writer = csv.writer(f_sim_log, lineterminator='\n')
            writer.writerow(['timestamp', "target_send_rate", "send_rate",
                             'recv_rate', 'latency',
                             'loss', 'reward', "action", "bytes_sent",
                             "bytes_acked", "bytes_lost", "MI",
                             "send_start_time",
                             "send_end_time", 'recv_start_time',
                             'recv_end_time', 'latency_increase',
                             "packet_size", 'min_lat', 'sent_latency_inflation',
                             'latency_ratio', 'send_ratio',
                             'bandwidth', "queue_delay",
                             'packet_in_queue', 'queue_size', "recv_ratio", "srtt"])
        else:
            f_sim_log = None
            writer = None
        test_scheduler = TestScheduler(trace)
        env = gym.make('AuroraEnv-v0', trace_scheduler=test_scheduler,
                       record_pkt_log=self.record_pkt_log)
        env.seed(self.seed)
        obs = env.reset()
        num_steps = 0
        while True:
            if env.net.senders[0].got_data:
                action = self.model.predict(obs)
            else:
                action = np.array([0])

            # get the new MI and stats collected in the MI
            # sender_mi = env.senders[0].get_run_data()
            sender_mi = env.senders[0].history.back() #get_run_data()
            throughput = sender_mi.get("recv rate")  # bits/sec
            send_rate = sender_mi.get("send rate")  # bits/sec
            latency = sender_mi.get("avg latency")
            loss = sender_mi.get("loss ratio")
            avg_queue_delay = sender_mi.get('avg queue delay')
            sent_latency_inflation = sender_mi.get('sent latency inflation')
            latency_ratio = sender_mi.get('latency ratio')
            send_ratio = sender_mi.get('send ratio')
            recv_ratio = sender_mi.get('recv ratio')
            reward = pcc_aurora_reward(
                throughput / BITS_PER_BYTE / BYTES_PER_PACKET, latency, loss,
                trace.avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET,
                trace.avg_delay * 2 / 1e3)
            if save_dir and writer:
                writer.writerow([
                    round(env.net.get_cur_time(), 6), round(env.senders[0].pacing_rate * BITS_PER_BYTE, 0),
                    round(send_rate, 0), round(throughput, 0), round(latency, 6), loss,
                    round(reward, 4), action.item(), sender_mi.bytes_sent, sender_mi.bytes_acked,
                    sender_mi.bytes_lost, round(sender_mi.send_end, 6) - round(sender_mi.send_start, 6),
                    round(sender_mi.send_start, 6), round(sender_mi.send_end, 6),
                    round(sender_mi.recv_start, 6), round(sender_mi.recv_end, 6),
                    sender_mi.get('latency increase'), sender_mi.packet_size,
                    sender_mi.get('conn min latency'), sent_latency_inflation,
                    latency_ratio, send_ratio,
                    env.links[0].get_bandwidth(
                        env.net.get_cur_time()) * BYTES_PER_PACKET * BITS_PER_BYTE,
                    avg_queue_delay, env.links[0].pkt_in_queue, env.links[0].queue_size,
                    recv_ratio, env.senders[0].srtt])
            reward_list.append(reward)
            loss_list.append(loss)
            delay_list.append(latency * 1000)
            tput_list.append(throughput / 1e6)
            send_rate_list.append(send_rate / 1e6)
            ts_list.append(env.net.get_cur_time())
            action_list.append(action.item())
            mi_list.append(sender_mi.send_end - sender_mi.send_start)
            obs_list.append(obs.tolist())
            obs, rewards, dones, info = env.step(action.item())
            num_steps += 1

            if dones or num_steps >= MAX_TIMESTEPS_PER_EPISODE:
                break
        if f_sim_log:
            f_sim_log.close()
        if self.record_pkt_log and save_dir:
            with open(os.path.join(save_dir, "{}_packet_log.csv".format(self.cc_name)), 'w', 1) as f:
                pkt_logger = csv.writer(f, lineterminator='\n')
                pkt_logger.writerow(['timestamp', 'packet_event_id', 'event_type',
                                     'bytes', 'cur_latency', 'queue_delay',
                                     'packet_in_queue', 'sending_rate', 'bandwidth'])
                pkt_logger.writerows(env.net.pkt_log)
        avg_sending_rate = env.senders[0].avg_sending_rate
        tput = env.senders[0].avg_throughput
        avg_lat = env.senders[0].avg_latency
        loss = env.senders[0].pkt_loss_rate
        pkt_level_reward = pcc_aurora_reward(tput, avg_lat,loss,
            avg_bw=trace.avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET)
        pkt_level_original_reward = pcc_aurora_reward(tput, avg_lat, loss)
        if plot_flag and save_dir:
            plot_simulation_log(trace, os.path.join(save_dir, '{}_simulation_log.csv'.format(self.cc_name)), save_dir, self.cc_name)
            bin_tput_ts, bin_tput = env.senders[0].bin_tput
            bin_sending_rate_ts, bin_sending_rate = env.senders[0].bin_sending_rate
            lat_ts, lat = env.senders[0].latencies
            plot(trace, bin_tput_ts, bin_tput, bin_sending_rate_ts,
                 bin_sending_rate, tput * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6,
                 avg_sending_rate * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6,
                 lat_ts, lat, avg_lat * 1000, loss, pkt_level_original_reward,
                 pkt_level_reward, save_dir, self.cc_name)
        if save_dir:
            with open(os.path.join(save_dir, "{}_summary.csv".format(self.cc_name)), 'w', 1) as f:
                summary_writer = csv.writer(f, lineterminator='\n')
                summary_writer.writerow([
                    'trace_average_bandwidth', 'trace_average_latency',
                    'average_sending_rate', 'average_throughput',
                    'average_latency', 'loss_rate', 'mi_level_reward',
                    'pkt_level_reward'])
                summary_writer.writerow(
                    [trace.avg_bw, trace.avg_delay,
                     avg_sending_rate * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6,
                     tput * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6, avg_lat,
                     loss, np.mean(reward_list), pkt_level_reward])

        return ts_list, reward_list, loss_list, tput_list, delay_list, send_rate_list, action_list, obs_list, mi_list, pkt_level_reward

    def test(self, trace: Trace, save_dir: str, plot_flag: bool = False, saliency: bool = False) -> Tuple[float, float]:
        _, reward_list, _, _, _, _, _, _, _, pkt_level_reward = self._test(trace, save_dir, plot_flag, saliency)
        return np.mean(reward_list), pkt_level_reward

def test_on_trace(model_path: str, trace: Trace, save_dir: str, seed: int,
                  record_pkt_log: bool = False, plot_flag: bool = False, enable_meta: bool = False):
    rl = Aurora(seed=seed, log_dir="", pretrained_model_path=model_path,
                timesteps_per_actorbatch=10, record_pkt_log=record_pkt_log, verbose=False, enable_meta=enable_meta)
    return rl.test(trace, save_dir, plot_flag)

def test_on_traces(model_path: str, traces: List[Trace], save_dirs: List[str],
                   nproc: int, seed: int, record_pkt_log: bool, plot_flag: bool, enable_meta: bool = False):
    arguments = [(model_path, trace, save_dir, seed, record_pkt_log, plot_flag, enable_meta)
                 for trace, save_dir in zip(traces, save_dirs)]
    with mp.Pool(processes=nproc) as pool:
        results = pool.starmap(test_on_trace, tqdm.tqdm(arguments, total=len(arguments)))
    return results
