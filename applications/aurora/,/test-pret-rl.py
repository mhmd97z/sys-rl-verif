from aurora_lib.network_simulator.pcc.aurora.schedulers import TestScheduler
from aurora_lib.network_simulator.pcc.aurora.aurora import Aurora
from aurora_lib.trace import generate_trace
import gym
from aurora_lib.ppo import PPO

# config_file_path = "config/udr.json"
# training_traces = []
# train_scheduler = UDRTrainScheduler(
#     config_file_path,
#     training_traces,
#     percent=0.0#args.real_trace_prob,
# )

# env = gym.make('AuroraEnv-v0', trace_scheduler=train_scheduler)
dummy_trace = generate_trace(
    (10, 10), (2, 2), (2, 2), (50, 50), (0, 0), (1, 1), (0, 0), (0, 0))
test_scheduler = TestScheduler(dummy_trace)
env = gym.make('AuroraEnv-v0', trace_scheduler=test_scheduler)
env.seed(20)
obs = env.reset()
print("The initial observation is {}".format(obs), type(obs))
model = PPO(env, verbose=False)