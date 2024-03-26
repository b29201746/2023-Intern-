# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:07:11 2022

@author: YC
"""

import numpy as np
from scipy import signal


def sig_filter(sig_i, FILTER_A, FILTER_B, input_prev, output_prev):
    filter_length = 4
    buf = FILTER_B[0] * sig_i
    for j in range(1, filter_length + 1):
        buf += input_prev[j - 1] * FILTER_B[j]
        buf -= output_prev[j - 1] * FILTER_A[j]

    input_prev[1:] = input_prev[:-1]
    output_prev[1:] = output_prev[:-1]
    input_prev[0] = sig_i
    output_prev[0] = buf

    return buf


def merge_interval(
    interval, prev_inter_arr, merge, merge_val, inter_thr, prev_interval, am1, am2
):
    inter1 = prev_inter_arr[1] - prev_inter_arr[0]
    inter2 = interval - prev_inter_arr[1]
    new_interval = prev_inter_arr[1] + interval
    if merge:
        if (
            merge_val * 0.9 < (prev_inter_arr[1] + interval) < merge_val * 1.1
            and abs(inter1 + inter2) <= 2
        ):
            interval += prev_inter_arr[1]
            merge_val = merge_val * 0.75 + interval * 0.25
    else:
        am_ratio = min([am1, am2]) / max([am1, am2])
        inter_ratio = min([prev_inter_arr[1], interval]) / max(
            [prev_inter_arr[1], interval]
        )
        if (
            (am_ratio < 0.6 or interval <= inter_thr)
            and inter_ratio > 0.7
            and new_interval < inter_thr * 2.5
        ):
            if abs(inter1 + inter2) <= 2 or (
                prev_interval * 0.85 < new_interval < prev_interval * 1.15
                if prev_interval is not None
                else False
            ):
                merge = True
                interval += prev_inter_arr[1]
                merge_val = interval

    return merge, merge_val, interval


def cal_cadence_time(sig, mode, prev_candence=None, fs=25):
    """
    calculate cadence(spm-steps pre min) based on time domain algo

    Parameters
    ----------
    sig : 1-dim float array
        L2-norm acc signal(unit:mg)
    mode : string
        "run" or "walk" mode
    prev_candence : float, optional
        previous window/sec cadence. The default is None.
    fs : float, optional
        sampling rate of acc signal. The default is 25.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    sig_len = len(sig)
    sig_std = np.mean([np.std(sig[i : i + fs]) for i in np.arange(0, sig_len, fs)])

    if mode == "walk":
        # order2 bandpass cutoff:0.5-1.25Hz 25Hz
        # FILTER_B, FILTER_A = signal.butter(2, [0.6, 1.15], "bandpass", fs=fs)
        FILTER_B = [0.00434586, 0.0, -0.00869172, 0.0, 0.00434586]
        FILTER_A = [1.0, -3.72230119, 5.27647689, -3.37489568, 0.82243563]
    else:
        # order2 bandpass cutoff:1-4Hz 25Hz
        # FILTER_B, FILTER_A = signal.butter(2, [1.5, 4], "bandpass", fs=fs)
        FILTER_B = [0.0913149, 0.0, -0.1826298, 0.0, 0.0913149]
        FILTER_A = [1.0, -2.59505054, 2.7465227, -1.45983194, 0.34766539]

    if prev_candence is not None:
        prev_interval = (
            fs / (prev_candence / 2 / 60)
            if mode == "walk"
            else fs / (prev_candence / 60)
        )
    else:
        prev_interval = None

    wn1 = int(fs * 0.8) if mode == "walk" else int(fs * 0.5)
    wn2 = int(fs * 0.4) if mode == "walk" else int(fs * 0.25)
    sig_f = []

    buffer_len = wn1
    mid_idx = int(buffer_len / 2)
    sig_buffer = np.zeros(buffer_len)

    locmax_list = []
    locmin_list = []
    all_interval = []
    prev_min_inter = np.zeros(2)
    prev_max_inter = np.zeros(2)

    cycle_idx = 0
    k = 0
    value_fill = 0
    input_prev = np.zeros(4)
    output_prev = np.zeros(4)
    ma1_list = []
    ma2_list = []

    interval = -1
    prev_locmax = -1
    prev_locmin = -1
    prev_locmax_val = -1
    prev_locmin_val = -1
    merge = False
    merge_val = None

    ma1_sum = 0
    ma2_sum = 0
    ma1 = 0
    ma2 = 0
    val = 0
    idx = 0
    curr = 0
    prev = 0
    wn1_half = int(wn1 / 2)
    wn2_half = int(wn2 / 2)

    lowerlim = fs * 0.8 if mode == "walk" else fs * 0.25
    upperlim = fs * 2 if mode == "walk" else fs

    sig_mean = np.mean(sig)
    sig_std = np.std(sig)
    curr_sig = 0
    # print(sig[0])
    for i in range(0, sig_len):
        # cal moving average with window size wn1,wn2(wn1 > wn2)
        if i == 0:
            value_fill = sig_filter(
                (sig[k] - sig_mean) / sig_std,
                FILTER_A,
                FILTER_B,
                input_prev,
                output_prev,
            )
            k += 1
            for j in range(buffer_len):
                if j <= mid_idx:
                    sig_buffer[j] = value_fill
                else:
                    sig_buffer[j] = sig_filter(
                        (sig[k] - sig_mean) / sig_std,
                        FILTER_A,
                        FILTER_B,
                        input_prev,
                        output_prev,
                    )
                    k += 1
            for j in range(-wn1_half, (wn1 - wn1_half)):
                ma1_sum += sig_buffer[mid_idx + j]

            for j in range(-wn2_half, (wn2 - wn2_half)):
                ma2_sum += sig_buffer[mid_idx + j]

            cycle_idx = buffer_len - 1
        else:
            if mid_idx - wn1_half >= 0:
                ma1_sum -= sig_buffer[mid_idx - wn1_half]
            else:
                ma1_sum -= sig_buffer[mid_idx + buffer_len - wn1_half]

            if mid_idx - wn2_half >= 0:
                ma2_sum -= sig_buffer[mid_idx - wn2_half]
            else:
                ma2_sum -= sig_buffer[mid_idx + buffer_len - wn2_half]

            if k < sig_len:
                cycle_idx = cycle_idx + 1 if cycle_idx + 1 < buffer_len else 0
                sig_buffer[cycle_idx] = sig_filter(
                    (sig[k] - sig_mean) / sig_std,
                    FILTER_A,
                    FILTER_B,
                    input_prev,
                    output_prev,
                )
                if k == sig_len - 1:
                    value_fill = sig_buffer[cycle_idx]
                k += 1
            else:
                cycle_idx = cycle_idx + 1 if cycle_idx + 1 < buffer_len else 0
                sig_buffer[cycle_idx] = value_fill

            mid_idx = mid_idx + 1 if mid_idx + 1 < buffer_len else 0

            if mid_idx + (wn1 - wn1_half) - 1 < buffer_len:
                ma1_sum += sig_buffer[mid_idx + (wn1 - wn1_half) - 1]
            else:
                ma1_sum += sig_buffer[mid_idx + (wn1 - wn1_half) - 1 - buffer_len]

            if mid_idx + (wn2 - wn2_half) - 1 < buffer_len:
                ma2_sum += sig_buffer[mid_idx + (wn2 - wn2_half) - 1]
            else:
                ma2_sum += sig_buffer[mid_idx + (wn2 - wn2_half) - 1 - buffer_len]

        curr_sig = sig_buffer[mid_idx]
        sig_f.append(curr_sig)

        ma1 = ma1_sum / wn1
        ma1_list.append(ma1)
        ma2 = ma2_sum / wn2
        ma2_list.append(ma2)

        # if curr is 1, find local maximum else find local minimum
        if ma2 > ma1:
            curr = 1
        else:
            curr = 0

        if i == 0:
            val = curr_sig
            idx = i
        else:
            if prev != curr:
                if prev == 1:
                    if prev_locmax > 0:
                        interval = idx - prev_locmax

                        if mode == "walk":
                            merge, merge_val, interval = merge_interval(
                                interval,
                                prev_max_inter,
                                merge,
                                merge_val,
                                (lowerlim * 0.8),
                                prev_interval,
                                (prev_locmax_val - prev_locmin_val),
                                (val - prev_locmin_val),
                            )
                        prev_max_inter[0] = prev_max_inter[1]
                        prev_max_inter[1] = idx - prev_locmax

                    locmax_list.append(idx)
                    prev_locmax = idx
                    prev_locmax_val = val

                else:
                    if prev_locmin > 0:
                        interval = idx - prev_locmin

                        if mode == "walk":
                            merge, merge_val, interval = merge_interval(
                                interval,
                                prev_min_inter,
                                merge,
                                merge_val,
                                (lowerlim * 0.8),
                                prev_interval,
                                (prev_locmax_val - prev_locmin_val),
                                (prev_locmax_val - val),
                            )
                        prev_min_inter[0] = prev_min_inter[1]
                        prev_min_inter[1] = idx - prev_locmin

                    locmin_list.append(idx)
                    prev_locmin = idx
                    prev_locmin_val = val

                if interval >= lowerlim and interval <= upperlim:
                    all_interval.append(interval)

                val = curr_sig
                idx = i
            else:
                if curr == 1 and curr_sig > val:
                    val = curr_sig
                    idx = i
                elif curr == 0 and curr_sig < val:
                    val = curr_sig
                    idx = i
        prev = curr

    # print(locmax_list)
    # print(locmin_list)
    # print(merge)
    # print(len(all_interval), np.sort(all_interval))
    if merge and len(all_interval) >= 3:
        all_interval = np.array(all_interval)[all_interval >= merge_val * 0.7]

    cadence = None
    confidence = None
    if len(all_interval) >= 3:
        all_interval = np.sort(all_interval)
        med_interval = np.median(all_interval)
        Q1_idx = int(0.25 * len(all_interval))
        Q3_idx = int(0.75 * len(all_interval))
        mean_interval = np.mean(all_interval[Q1_idx : Q3_idx + 1])

        # confidence
        score1 = (all_interval[Q3_idx] - all_interval[Q1_idx]) / med_interval
        score1 = 1 if score1 / 0.5 > 1 else (1 - score1 / 0.5)
        score2 = len(all_interval) / (sig_len * 2 / mean_interval)
        score2 = 1 if score2 > 1 else score2
        score3 = abs(mean_interval - med_interval) / (mean_interval + med_interval)
        score3 = 1 if score3 / 0.05 > 1 else (1 - score3 / 0.05)
        confidence = score1 * 0.2 + score2 * 0.6 + score3 * 0.2

        if (all_interval[Q3_idx] - all_interval[Q1_idx]) / med_interval < 0.3:
            if mode == "walk":
                cadence = fs / mean_interval * 60 * 2
            else:
                cadence = fs / mean_interval * 60
        else:
            if mode == "walk":
                cadence = fs / med_interval * 60 * 2
            else:
                cadence = fs / med_interval * 60

    return cadence, confidence
