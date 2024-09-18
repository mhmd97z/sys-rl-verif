import csv
import os
from typing import List, Tuple
import argparse

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, '../')
sys.path.insert(0, './')

from utils import compute_std_of_mean, load_summary, pcc_aurora_reward
from trace import Trace
from synthetic_dataset import SyntheticDataset
from network_simulator.constants import (BITS_PER_BYTE, BYTES_PER_PACKET)


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Plotting Aurora model testing in simulator with synthetic datasets.")
    parser.add_argument('--save-dir', type=str, default="",
                        help="direcotry to testing results.")
    # parser.add_argument('--dataset-dir', type=str, default="data/synthetic_dataset",
    #                     help="direcotry to dataset.")

    args, _ = parser.parse_known_args()
    return args


def load_summaries_across_traces(log_files: List[str]) -> Tuple[List[float], List[float], List[float], List[float]]:
    rewards = []
    tputs = []
    lats = []
    losses = []
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(log_file, 'does not exist!')
            continue
        summary = load_summary(log_file)
        rewards.append(summary['pkt_level_reward'])
        tputs.append(summary['average_throughput'])
        lats.append(summary['average_latency'] * 1000)
        losses.append(summary['loss_rate'])
        # rewards.append(summary['mi_level_reward'])
        # rewards.append(pcc_aurora_reward(summary['average_throughput'] * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET,
        #                summary['average_latency'], summary['loss_rate']))
    return rewards, tputs, lats, losses


def load_results(save_dirs, seeds, steps):
    rewards, tputs, lats, losses = [], [], [], []
    reward_errs, tput_errs, lat_errs, loss_errs = [], [], [], []
    for step in steps:
        step = int(step)
        avg_rewards_across_seeds = []
        avg_tputs_across_seeds = []
        avg_lats_across_seeds = []
        avg_losses_across_seeds = []

        for seed in seeds:
            tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = \
                    load_summaries_across_traces([os.path.join(
                save_dir, 'aurora_summary.csv') for save_dir in save_dirs])
            avg_rewards_across_seeds.append(np.nanmean(np.array(tmp_rewards)))
            avg_tputs_across_seeds.append(np.nanmean(np.array(tmp_tputs)))
            avg_lats_across_seeds.append(np.nanmean(np.array(tmp_lats)))
            avg_losses_across_seeds.append(np.nanmean(np.array(tmp_losses)))

        # print(name, avg_rewards_across_seeds)
        rewards.append(np.nanmean(np.array(avg_rewards_across_seeds)))
        reward_errs.append(compute_std_of_mean(avg_rewards_across_seeds))

        tputs.append(np.nanmean(np.array(avg_tputs_across_seeds)))
        tput_errs.append(compute_std_of_mean(avg_tputs_across_seeds))

        lats.append(np.nanmean(np.array(avg_lats_across_seeds)))
        lat_errs.append(compute_std_of_mean(avg_lats_across_seeds))

        losses.append(np.nanmean(np.array(avg_losses_across_seeds)))
        loss_errs.append(compute_std_of_mean(avg_losses_across_seeds))

    low_bnd = np.array(rewards) - np.array(reward_errs)
    up_bnd = np.array(rewards) + np.array(reward_errs)

    tputs_low_bnd = np.array(tputs) - np.array(tput_errs)
    tputs_up_bnd = np.array(tputs) + np.array(tput_errs)

    lats_low_bnd = np.array(lats) - np.array(lat_errs)
    lats_up_bnd = np.array(lats) + np.array(lat_errs)

    losses_low_bnd = np.array(losses) - np.array(loss_errs)
    losses_up_bnd = np.array(losses) + np.array(loss_errs)
    return (rewards, tputs, lats, losses, low_bnd, up_bnd,
            tputs_low_bnd, tputs_up_bnd, lats_low_bnd, lats_up_bnd,
            losses_low_bnd, losses_up_bnd)


def main():
    args = parse_args()
    # dataset = SyntheticDataset.load_from_dir(args.dataset_dir)
    count = len(next(os.walk(args.save_dir))[1])
    save_dirs = [os.path.join(args.save_dir, 'trace_{:05d}'.format(i)) for i in range(count)]
    seeds = list(range(10, 31, 10))

    steps = [720000]
    rewards, tputs, lats, losses, \
    low_bnd, up_bnd, \
    tputs_low_bnd, tputs_up_bnd, \
    lats_low_bnd, lats_up_bnd, \
    losses_low_bnd, losses_up_bnd = load_results(save_dirs, seeds, steps)
    rewards_err = (up_bnd[0] - low_bnd[0]) / 2

    print('Reward:', rewards, low_bnd, up_bnd)
    print('Reward Error:', rewards_err)
    print('Throughput:', tputs, tputs_low_bnd, tputs_up_bnd)
    print('Latency:', lats, lats_low_bnd, lats_up_bnd)
    print('Loss:', losses, losses_low_bnd, losses_up_bnd)


if __name__ == "__main__":
    main()
