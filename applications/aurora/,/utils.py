import numpy as np
import random
import json
import os
import random
from typing import Dict, Optional
import csv

"""
Hyperparameters
"""
DISCOUNT = 0.99  # 0.8 to 0.9997 (most common: 0.99)
HIDDEN_SIZE = 64  # hidden layer width
LR = 3e-4  # 5e-3 5e-6 [0.003, 5e-6]
SGD_EPOCHS = 5  # [3, 30]
MINI_BATCH_SIZE = 5  # 4 to 4096 (can be much higher with distributed implementations)
CLIP = 0.2  # 0.1, 0.2, 0.3
ENTROPY_COEFFICIENT = 0.01  # 0.001 [0, 0.01]
CRITIC_LOSS_DISCOUNT = 0.05  # 0.03 0.05 [0.5, 1]

MAX_NUM_TIMESTEPS = 100000000
TOTAL_ITERATIONS = 10240
EPISODES_PER_ITERATION = 10  # 5 10 15 20
MAX_TIMESTEPS_PER_EPISODE = 32
MAX_SAME_ITERATIONS = 2 * SGD_EPOCHS

FLAG_CONTINUOUS_ACTION = True
CHECKPOINT_DIR = './checkpoints/'
LOG_DIR = './logs/'
SAVE_TO_FILE = False

"""
Meta-learning
"""
BUFFER_UPDATE_MODE = 'best'
BUFFER_SIZE = 32

NUM_FEATURES_PER_SHOT = 32  # 30 (obs) + 1 (action) + 1 (reward)
RNN_HIDDEN_SIZE = 32  # equal to the max steps of each episode
RNN_NUM_LAYERS = 2
EMBEDDING_DIM = 32  # 32


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def save_args(args, save_dir: str):
    """Write arguments to a log file."""
    os.makedirs(save_dir, exist_ok=True)
    if save_dir and os.path.exists(save_dir):
        write_json_file(os.path.join(save_dir, 'cmd.json'), args.__dict__)

def read_json_file(filename):
    """Load json object from a file."""
    with open(filename, 'r') as f:
        content = json.load(f)
    return content

def write_json_file(filename, content):
    """Dump into a json file."""
    with open(filename, 'w') as f:
        json.dump(content, f, indent=4)

def pcc_aurora_reward(throughput: float, delay: float, loss: float,
                      avg_bw: Optional[float] = None,
                      min_rtt: Optional[float] = None) -> float:
    """PCC Aurora reward. Anchor point 0.6Mbps
    throughput: packets per second
    delay: second
    loss:
    avg_bw: packets per second
    """
    # if avg_bw is not None and min_rtt is not None:
    #     return 10 * 50 * throughput/avg_bw - 1000 * delay * 0.2 / min_rtt - 2000 * loss
    if avg_bw is not None:
        return 10 * 50 * throughput/avg_bw - 1000 * delay - 2000 * loss
    return 10 * throughput - 1000 * delay - 2000 * loss

def zero_one_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def load_summary(summary_file: str) -> Dict[str, float]:
    summary = {}
    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                summary[k] = float(v)
    return summary

def compute_std_of_mean(data):
    return np.std(data) / np.sqrt(len(data))
