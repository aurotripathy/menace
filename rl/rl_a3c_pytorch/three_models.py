import subprocess
import time

cmd_start_list = "python3.6 gym-matplotlib-animated-eval.py \
--env MsPacman-v0 \
--render True \
--train-time 50".split()

print(cmd_start_list)

# cmd_stop = "pkill python"

train_list = [50, 100, 200]
sleep_list = [10, 15, 20]
while True:
    for train_time, sleep_time in zip(train_list, sleep_list): 
        print("In Loop...")
        cmd_start_list[-1] = str(train_time) # replace default
        sub_p = subprocess.Popen(cmd_start_list)
        print("process pid:", sub_p.pid)
        print("Starting sleep")
        time.sleep(sleep_time)
        print("Done sleep")
        sub_p.kill()

print("Done")
