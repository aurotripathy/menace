import subprocess
import time

cmd_start_list = "python3.6 gym-matplotlib-animated-eval.py \
--env MsPacman-v0 \
--render True".split()

print(cmd_start_list)

cmd_stop = "pkill python"

while True:
    print("In Loop...")
    sub_p = subprocess.Popen(cmd_start_list)
    print("process pid:", sub_p.pid)
    print("Starting sleep")
    time.sleep(15)
    print("Done sleep")
    sub_p.kill()

print("Done")
