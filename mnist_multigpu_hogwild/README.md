`python main.py --cuda --num-processes 20 --gpu-ids 0 1 2 3`

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.40.04    Driver Version: 418.40.04    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:1B.0 Off |                    0 |
| N/A   54C    P0    58W / 300W |   5116MiB / 16130MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  On   | 00000000:00:1C.0 Off |                    0 |
| N/A   51C    P0    56W / 300W |   5116MiB / 16130MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  On   | 00000000:00:1D.0 Off |                    0 |
| N/A   49C    P0    59W / 300W |   5116MiB / 16130MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |
| N/A   51C    P0    60W / 300W |   5116MiB / 16130MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     50341      C   Training Agent: 0                           1021MiB |
|    0     50345      C   Training Agent: 4                           1021MiB |
|    0     50349      C   Training Agent: 8                           1021MiB |
|    0     50353      C   Training Agent: 12                          1021MiB |
|    0     50357      C   Training Agent: 16                          1021MiB |
|    1     50342      C   Training Agent: 1                           1021MiB |
|    1     50346      C   Training Agent: 5                           1021MiB |
|    1     50350      C   Training Agent: 9                           1021MiB |
|    1     50354      C   Training Agent: 13                          1021MiB |
|    1     50358      C   Training Agent: 17                          1021MiB |
|    2     50343      C   Training Agent: 2                           1021MiB |
|    2     50347      C   Training Agent: 6                           1021MiB |
|    2     50351      C   Training Agent: 10                          1021MiB |
|    2     50355      C   Training Agent: 14                          1021MiB |
|    2     50359      C   Training Agent: 18                          1021MiB |
|    3     50344      C   Training Agent: 3                           1021MiB |
|    3     50348      C   Training Agent: 7                           1021MiB |
|    3     50352      C   Training Agent: 11                          1021MiB |
|    3     50356      C   Training Agent: 15                          1021MiB |
|    3     50360      C   Training Agent: 19                          1021MiB |
+-----------------------------------------------------------------------------+
```
