import sys
import numpy as np


def set_hyperparams():
    lstm_size = 320
    learning_rate = 1e-3
    seq_len = 100
    batch_size = 64
    batches = 500
    return lstm_size, learning_rate, seq_len, batch_size, batches


def get_batch(seed=11, shape=(100, 64, 125), classes=10):
    _, batch_size, _ = shape
    np.random.seed(seed)

    # Samples
    bX = np.float32(np.random.uniform(-1, 1, (shape)))

    # Targets
    bY = np.int32(np.random.randint(low=0, high=classes - 1, size=batch_size))

    return bX, bY



def print_results(batch_loss_list, batch_time_list, train_start, train_end):

    abort = False

    # 0. Check if loss is numeric (not NAN and not inf)
    check_loss = [np.isfinite(loss) for loss in batch_loss_list]
    if False not in check_loss:
        print('>>> Loss check 1/2 passed: loss is finite {}'.format(np.unique(check_loss)))
    else:
        print('!!! Loss check 1/2 failed: loss is NOT finite {}'.format(np.unique(check_loss)))
        abort = True

    # 1. Check if loss is decreasing
    check_loss = np.diff(batch_loss_list)
    if np.sum(check_loss) < 0:
        print('>>> Loss check 2/2 passed: loss is globally decreasing')
    else:
        print('!!! Loss check 2/2 failed: loss is NOT globally decreasing')
        abort = True

    # 2. Check deviation between the full loop time and the sum of individual batches
    loop_time = train_end - train_start
    batch_time_sum = np.sum(batch_time_list)
    factor = loop_time / batch_time_sum
    deviation = np.abs((1 - factor) * 100)

    if deviation < 1:  # Less than 1% deviation
        print('>>> Timing check passed -  < 1% deviation between loop time and sum of batches')
        
        print("Total time {:.3f} seconds. Sum of batch times {:.3f} seconds. Deviation [%] {:.3f}".format(loop_time,
                                                                                                          batch_time_sum,
                                                                                                          deviation))
    else:
        print('!!! Timing check failed - Deviation > 1% ::: Loop time {:.3f} ::: Sum of batch times {:.3f} :::'
	      ' Deviation [%] {:.3f}'.format(loop_time, batch_time_sum, deviation))
        abort = True

    if abort:
        sys.exit('!!! Abort benchmark.')
        print('=' * 100)
