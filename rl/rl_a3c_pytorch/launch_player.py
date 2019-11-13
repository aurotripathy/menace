import subprocess
from multiprocessing.connection import Client
import time
import argparse

class LaunchPlayer():
    def __init__(self):
        self.cmd = "python3.6 gym-matplotlib-animated-eval.py \
        --env MsPacman-v0 \
        --render True \
        --train-time 50 \
        --load-model-dir /dockerx/data/rl/trained_models/".split()

        self.running_process = None
        self.train_times = [53, 150, 550]
        self.play_times = [15, 30, 50]
        self.model_locations = ['/dockerx/data/rl/trained_models-53m/',
                                '/dockerx/data/rl/trained_models-150m/',
                                '/dockerx/data/rl/trained_models-550m/',]

    def play_all_models(self):

        print(self.cmd)

        address = ('localhost', 6000)
        conn = Client(address, authkey=str.encode('sc19-visuals'))

        while True:
            for train_time, play_time, model_location in zip(self.train_times,
                                              self.play_times, self.model_locations): 
                conn.send('next')
                print("Send request to diplay server...")
                self.cmd[-3] = str(train_time) # replace default
                self.cmd[-1] = model_location
                self.running_process = subprocess.Popen(self.cmd)
                print("process pid:", self.running_process.pid)
                print("Starting sleep")
                time.sleep(play_time)
                print("Done sleep")
                self.running_process.kill()

    def play_checkpointed_model(self, checkpoint_num):
        train_time = self.train_times[checkpoint_num]
        model_location = self.model_locations[checkpoint_num]
        self.cmd[-3] = str(train_time) # replace default
        self.cmd[-1] = model_location
        self.running_process = subprocess.Popen(self.cmd)
        print("process pid:", self.running_process.pid)


    def exit_player(self):
        self.running_process.kill()
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pick the checkpoint to play.')
    parser.add_argument(
        '--control-option',
        type=int,
        metavar='OPT',
        help='The control option to play')
    args = parser.parse_args()
    player = LaunchPlayer()
    if args.control_option == 3:
        player.play_all_models()
    elif args.control_option in range(len(player.model_locations)):
        player.play_checkpointed_model(args.control_option)
    else:
        print('Bad checkpoint number')
    print('Done launchng!')
    time.sleep(5)
    player.exit_player()
