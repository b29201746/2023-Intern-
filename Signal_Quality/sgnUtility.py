import numpy as np
import pandas as pd
import pickle

FS_PPG_OH1 = 135
FS_ACC_OH1 = 51


def loadPickle(path):
    with open(path, "rb") as f:
        new_dict = pickle.load(f)
    return new_dict


def savePickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def resampleSGN(signal, fsIn, fsOut):
    if fsIn == fsOut:
        return signal.copy()
    new_signal_size = int(len(signal) / fsIn * fsOut)
    interval = fsIn / fsOut
    last_pt, next_pt = 0, 0
    interp_ratio, pt_diff, interp = 0, 0, 0
    output_signal = []
    for new_index in range(new_signal_size):
        last_pt = int(interval * new_index)
        next_pt = last_pt + 1 if last_pt + 1 < len(signal) else len(signal) - 1
        interp_ratio = interval * new_index - last_pt
        pt_diff = signal[next_pt] - signal[last_pt]
        interp = pt_diff * interp_ratio
        output_signal.append(signal[last_pt] + interp)
    return output_signal


def beatNormalization(beatSignal):
    """resample to 100 samples, normalize beat to 0-1 scale

    Args:
        beatSignal (list): 100 Hz beat
    """
    beatSignal = resampleSGN(beatSignal, len(beatSignal), 100)
    ymax = max(beatSignal)
    ymin = min(beatSignal)
    output = []
    for i in range(len(beatSignal)):
        output.append((beatSignal[i] - ymin) / (ymax - ymin))
    return output


def movingAverage(signal, winSize):
    result = []
    runSum = 0
    runPrevSum = 0
    for i in range(len(signal)):
        runSum += signal[i]
        if i < winSize:
            result.append(runSum / (i + 1))
        else:
            runPrevSum += signal[i - winSize]
            result.append((runSum - runPrevSum) / winSize)
    return result
