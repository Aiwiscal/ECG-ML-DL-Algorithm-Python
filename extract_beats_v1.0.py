#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import logging
import numpy as np
from biosppy.signals import ecg
from biosppy.storage import load_txt
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_extract_beats(data_path):
    signal, mdata = load_txt(data_path)
    logging.info("--------------------------------------------------")
    logging.info("载入信号-%s, 长度 = %d " % (data_path, len(signal)))
    fs = 360  # 信号采样率 360 Hz
    logging.info("调用 hamilton_segmenter 进行R波检测 ...")
    tic = time.time()
    rpeaks = ecg.hamilton_segmenter(signal, sampling_rate=fs)
    toc = time.time()
    logging.info("完成. 用时: %f 秒. " % (toc - tic))
    rpeaks = rpeaks[0]

    heart_rate = 60 / (np.mean(np.diff(rpeaks)) / fs)
    # np.diff 计算相邻R峰之间的距离分别有多少个采样点，np.mean求平均后，除以采样率，
    # 将单位转化为秒，然后计算60秒内可以有多少个RR间期作为心率
    logging.info("平均心率: %.3f / 分钟." % (heart_rate))

    win_before = 0.2
    win_after = 0.4
    logging.info("根据R波位置截取心拍, 心拍前窗口：%.3f 秒 ~ 心拍后窗口：%.3f 秒 ..." \
                 % (win_before, win_after))
    tic = time.time()
    beats, rpeaks_beats = ecg.extract_heartbeats(signal, rpeaks, fs, win_before, win_after)
    toc = time.time()
    logging.info("完成. 用时: %f 秒." % (toc - tic))
    logging.info("共截取到 %d 个心拍, 每个心拍长度为 %d 个采样点" % \
                 (beats.shape[0], beats.shape[1]))

    plt.figure()
    plt.grid(True)
    for i in range(beats.shape[0]):
        plt.plot(beats[i])
    plt.title(data_path)
    plt.show()
    return

if __name__ == '__main__':
    test_extract_beats("./data/ecg_records_103.txt")
    test_extract_beats("./data/ecg_records_117.txt")
    test_extract_beats("./data/ecg_records_119.txt")

