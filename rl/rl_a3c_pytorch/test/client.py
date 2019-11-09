#Your client could send commands as objects:

from multiprocessing.connection import Client
import time

address = ('localhost', 6000)
conn = Client(address, authkey=str.encode('sc19-visuals'))
while True:
    conn.send('next')
    time.sleep(3)
    # can also send arbitrary objects:
    # conn.send(['a', 2.5, None, int, sum])
# conn.close()
print('Client done')
