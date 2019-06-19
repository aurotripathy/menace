from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from environment import atari_env
from utils import read_config, setup_logger
from model import A3Clstm
from player_util import Agent
import gym
import logging
import time
from arg_parser import get_args
from pudb import set_trace
#from gym.configuration import undo_logger_setup

#undo_logger_setup()
from watchdog.observers import Observer
from watchdog.events import RegexMatchingEventHandler

saved_state = None  # Global scope, start w/None, init'd later 

class CheckPointHandler(RegexMatchingEventHandler):

    MODEL_REGEX = [r".*[^_thumbnail]\.dat$"]
    
    def __init__(self):
        super().__init__(self.MODEL_REGEX)
        
    def on_modified(self, event):
        print('Event:', event)
        self.process(event)

    def process(self, event):
        global saved_state
        print('Updating the model')
        saved_state = torch.load(
            '{0}{1}.dat'.format(args.load_model_dir, args.env),
            map_location=lambda storage, loc: storage)


class CheckPointWatcher:
    def __init__(self, src_path):
        self.__src_path = src_path
        self.__event_handler = CheckPointHandler()
        self.__event_observer = Observer()

    def run(self):
        self.start()
        try:
            do_normal_processing()
        except KeyboardInterrupt:
            self.stop()


    def start(self):
        self.__schedule()
        self.__event_observer.start()

    def stop(self):
        self.__event_observer.stop()
        self.__event_observer.join()

    def __schedule(self):
        self.__event_observer.schedule(
            self.__event_handler,
            self.__src_path,
            recursive=True
        )



def do_normal_processing():
    global saved_state
    
    gpu_id = args.gpu_id

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)

    while True:
        print('Looking for {}'.format(os.path.join(args.load_model_dir, args.env) + '.dat'))
        if os.path.exists(os.path.join(args.load_model_dir, args.env) + '.dat'):
            print("Found model...")
            saved_state = torch.load(
                '{0}{1}.dat'.format(args.load_model_dir, args.env),
                map_location=lambda storage, loc: storage)
            break
        else:
            print('...waiting for model to show up') 
        time.sleep(1)

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
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
    if args.new_gym_eval:
        player.env = gym.wrappers.Monitor(
            player.env, "{}_monitor".format(args.env), force=True)

    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model.load_state_dict(saved_state)
    else:
        player.model.load_state_dict(saved_state)

    player.model.eval()
    for i_episode in range(args.num_episodes):
        player.state = player.env.reset()
        player.state = torch.from_numpy(player.state).float()
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.state = player.state.cuda()
        player.eps_len += 2
        reward_sum = 0
        while True:
            if args.render:
                if i_episode % args.render_freq == 0:
                    player.env.render()

            player.action_test()
            reward_sum += player.reward

            if player.done and not player.info:
                state = player.env.reset()
                player.eps_len += 2
                player.state = torch.from_numpy(state).float()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = player.state.cuda()
            elif player.info:
                num_tests += 1
                reward_total_sum += reward_sum
                reward_mean = reward_total_sum / num_tests
                log['{}_mon_log'.format(args.env)].info(
                    "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)),
                        reward_sum, player.eps_len, reward_mean))
                player.eps_len = 0
                break

def get_env_conf():
    set_trace()
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    return env_conf

if __name__ == "__main__":
    args = get_args()
    env_conf = get_env_conf()
    CheckPointWatcher(args.load_model_dir).run()

            
