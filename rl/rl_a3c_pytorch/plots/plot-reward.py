""" 
Plots the average reward over time
The input is the log file from the output of the training
"""
import numpy as np
import pandas as pd
from pudb import set_trace
import matplotlib
import matplotlib.pyplot as plt

def pick_reward(str):
    return float(str.split(' ')[3])

def pick_time(str):
    time = str.split(' ')
    return time[3] + '.' + time[4]

df = pd.read_csv('trunc-MsPacman-v0_log',
                 delimiter=',',
                 names=['time_1', 'time_2', 'episode reward', 'episode len', 'reward mean'],
                 error_bad_lines=False)
df.reset_index(level=0, inplace=True)  # go to 0,1,... indexing

mean_rewards = df['reward mean'].apply(pick_reward)
time_strs = df['time_2'].apply(pick_time)
for mean_reward, time_str in zip(mean_rewards, time_strs):
    print(time_str, mean_reward)


# Plot
x = np.arange(len(time_strs))
my_xticks = time_strs
my_xticks = [my_xticks[i] if i%10==0 else ' ' for i,x in enumerate(my_xticks)]
plt.ylabel('Avg. reward')
plt.xticks(x, my_xticks, rotation='vertical')
plt.plot(x, mean_rewards)

plt.show()
plt.savefig('reward-over-time.png')
