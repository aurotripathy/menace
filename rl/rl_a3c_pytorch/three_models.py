

def process_in_sequence():
    import subprocess
    from multiprocessing.connection import Client
    import time

    cmd = "python3.6 gym-matplotlib-animated-eval.py \
    --env MsPacman-v0 \
    --render True \
    --train-time 50 \
    --load-model-dir /dockerx/data/rl/trained_models/".split()

    print(cmd)

    address = ('localhost', 6000)
    conn = Client(address, authkey=str.encode('sc19-visuals'))
    train_times = [53, 150, 550]
    play_times = [15, 30, 50]
    model_locations = ['/dockerx/data/rl/trained_models-53m/',
                       '/dockerx/data/rl/trained_models-150m/',
                       '/dockerx/data/rl/trained_models-550m/',]

    while True:
        for train_time, play_time, model_location in zip(train_times,
                                          play_times, model_locations): 
            conn.send('next')
            print("Send request to diplay server...")
            cmd[-3] = str(train_time) # replace default
            cmd[-1] = model_location
            sub_p = subprocess.Popen(cmd)
            print("process pid:", sub_p.pid)
            print("Starting sleep")
            time.sleep(play_time)
            print("Done sleep")
            sub_p.kill()


if __name__ == '__main__':
    process_in_sequence()
