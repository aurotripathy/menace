import subprocess
import time

cmd = "python3.6 gym-matplotlib-animated-eval.py \
--env MsPacman-v0 \
--render True \
--train-time 50 \
--load-model-dir /dockerx/data/rl/trained_models/".split()

print(cmd)

train_times = [10, 100, 1000]
sleep_times = [15, 20, 60]
model_locations = ['/dockerx/data/rl/trained_models/',
                   '/dockerx/data/rl/trained_models/',
                   '/dockerx/data/rl/trained_models/',]
while True:
    for train_time, sleep_time, model_location in zip(train_times,
                                      sleep_times, model_locations): 
        print("In Loop...")
        cmd[-3] = str(train_time) # replace default
        cmd[-1] = model_location
        sub_p = subprocess.Popen(cmd)
        print("process pid:", sub_p.pid)
        print("Starting sleep")
        time.sleep(sleep_time)
        print("Done sleep")
        sub_p.kill()


