import argparse

def get_eval_args():
    parser = argparse.ArgumentParser(description='A3C_EVAL')
    parser.add_argument(
        '--env',
        default='Pong-v0',
        metavar='ENV',
        help='environment to train on (default: Pong-v0)')
    parser.add_argument(
        '--env-config',
        default='config.json',
        metavar='EC',
        help='environment to crop and resize info (default: config.json)')
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=100,
        metavar='NE',
        help='how many episodes in evaluation (default: 100)')
    parser.add_argument(
        '--load-model-dir',
        default='trained_models/',
        metavar='LMD',
        help='folder to load trained models from')
    parser.add_argument(
        '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
    parser.add_argument(
        '--render',
        default=False,
        metavar='R',
        help='Watch game as it being played')
    parser.add_argument(
        '--render-freq',
        type=int,
        default=1,
        metavar='RF',
        help='Frequency to watch rendered game play')
    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=10000,
        metavar='M',
        help='maximum length of an episode (default: 100000)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=-1,
        help='GPU to use [-1 CPU only] (default: -1)')
    parser.add_argument(
        '--skip-rate',
        type=int,
        default=4,
        metavar='SR',
        help='frame skip rate (default: 4)')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (default: 1)')
    parser.add_argument(
        '--new-gym-eval',
        default=False,
        metavar='NGE',
        help='Create a gym evaluation for upload')
    return parser.parse_args()
