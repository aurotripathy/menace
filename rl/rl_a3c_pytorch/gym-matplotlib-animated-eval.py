from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
from eval_parser import get_eval_args
import torch
from environment import atari_env
from utils import read_config, setup_logger
from model import A3Clstm
from player_util import Agent
import gym
import logging
import time
from credits import credits_str, infomercial_str
import matplotlib.image as image
from plot_utils import place_figure

gpu_id = -1
args = get_eval_args()

setup_json = read_config(args.env_config)
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args.env:
        env_conf = setup_json[i]

torch.manual_seed(args.seed)

saved_state = torch.load(
    '{0}{1}.dat'.format(args.load_model_dir, args.env),
    map_location=lambda storage, loc: storage)

log = {}
setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
    args.log_dir, args.env))
log['{}_mon_log'.format(args.env)] = logging.getLogger('{}_mon_log'.format(
    args.env))

d_args = vars(args)
for k in d_args.keys():
    log['{}_mon_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

env = atari_env("{}".format(args.env), env_conf, args)
num_tests = 0
start_time = time.time()
reward_total_sum = 0
player = Agent(None, env, args, None)
player.model = A3Clstm(player.env.observation_space.shape[0],
                       player.env.action_space)
player.gpu_id = gpu_id
if args.new_gym_eval:
    player.env = gym.wrappers.Monitor(
        player.env, "{}_monitor".format(args.env), force=True)

player.model.load_state_dict(saved_state)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['toolbar'] = 'None'

def updatefig(*fargs):
    global num_tests
    global reward_total_sum
    global reward_sum
    global args

    im = plt.imshow(player.env.render(mode='rgb_array'), animated=True)


    player.action_test()  # versus _train
    reward_sum += player.reward

    if player.done and not player.info:
        state = player.env.reset()
        player.eps_len += 2
        player.state = torch.from_numpy(state).float()
    elif player.info:
        num_tests += 1
        reward_total_sum += reward_sum
        reward_mean = reward_total_sum / num_tests
        print("Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
              format(
                  time.strftime("%Hh %Mm %Ss",
                                time.gmtime(time.time() - start_time)),
                  reward_sum, player.eps_len, reward_mean))
        player.eps_len = 0

        # maybe another episode
        player.state = player.env.reset()
        player.state = torch.from_numpy(player.state).float()
        player.eps_len += 2
        reward_sum = 0
        
    return im,
    
fig = plt.figure(figsize=(7.1, 10.4), facecolor='black')
place_figure(fig, 1200, 10)
fig.canvas.set_window_title('Reinforcement Learning')
ax = fig.add_subplot(1, 1, 1, aspect=1)
# ax.set_title("PyTorch Plays Pacman", fontsize=20, verticalalignment='bottom', color='r')
ax.set_title("PyTorch Plays Pacman", fontsize=32, color='r')
footer_text = 'Model loaded at '
plt.text(-5, 218, footer_text,
         color='w', fontsize=20)
checkpoint_time = '{} minute mark'.format(args.train_time)
plt.text(65, 218, checkpoint_time,
         color='r', fontsize=20)

plt.text(-5, 237, infomercial_str,
         color='w', fontsize=16)

plt.text(-22, 257, credits_str,
         color='w', fontsize=8)

rocm_logo = image.imread('logos/ROCm_Logo_128.png')
epyc_logo = image.imread('logos/EPYC-blue-white.png')
radeon_logo = image.imread('logos/AMD_Radeon-white.png')

ax.figure.figimage(rocm_logo, 90, 900, alpha=1, zorder=1)
ax.figure.figimage(radeon_logo, 280, 935, alpha=1, zorder=1)
ax.figure.figimage(epyc_logo, 500, 925, alpha=1, zorder=1)

im = plt.imshow(env.render(mode='rgb_array'), animated=True)
        
player.model.eval()
player.state = player.env.reset()
player.state = torch.from_numpy(player.state).float()
player.eps_len += 2
reward_sum = 0
ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)
plt.show()
