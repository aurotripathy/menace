# https://stackoverflow.com/questions/6920858/interprocess-communication-in-python?rq=1

from multiprocessing.connection import Listener

address = ('localhost', 6000)     # family is deduced to be 'AF_INET'
listener = Listener(address, authkey=str.encode('sc19-visuals'))
conn = listener.accept()
print('connection accepted from', listener.last_accepted)
while True:
    msg = conn.recv()
    # do something with msg
    if msg == 'next':
        print("Got message", msg)
        # conn.close()
        # break
listener.close()
print('Listener done.')
