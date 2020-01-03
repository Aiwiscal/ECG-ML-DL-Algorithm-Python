#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import time
import logging
import numpy as np
from biosppy.signals import ecg
from biosppy.storage import load_txt
import matplotlib.pyplot as plt

logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rpeaks_simple(data_path):
    signal, mdata = load_txt(data_path)
    logging.info("--------------------------------------------------")
    logging.info("载入信号-%s, 长度 = %d " % (data_path, len(signal)))
    fs = 360  # 信号采样率 360 Hz
    logging.info("调用 christov_segmenter 进行R波检测 ...")
    tic = time.time()
    rpeaks = ecg.christov_segmenter(signal, sampling_rate=fs)
    toc = time.time()
    logging.info("完成. 用时: %f 秒. " % (toc - tic))
    # 以上这种方式返回的rpeaks类型为biosppy.utils.ReturnTuple, biosppy的内置类
    logging.info("直接调用 christov_segmenter 返回类型为 " + str(type(rpeaks)))

    # 得到R波位置序列的方法：
    # 1) 取返回值的第1项：
    logging.info("使用第1种方式取R波位置序列 ... ")
    rpeaks_indices_1 = rpeaks[0]
    logging.info("完成. 结果类型为 " + str(type(rpeaks_indices_1)))
    # 2) 调用ReturnTuple的as_dict()方法，得到Python有序字典（OrderedDict）类型
    logging.info("使用第2种方式取R波位置序列 ... ")
    rpeaks_indices_2 = rpeaks.as_dict()
    #    然后使用变量名（这里是rpeaks）作为key取值。
    rpeaks_indices_2 = rpeaks_indices_2["rpeaks"]
    logging.info("完成. 结果类型为 " + str(type(rpeaks_indices_2)))

    # 检验两种方法得到的结果是否相同：
    check_sum = np.sum(rpeaks_indices_1 == rpeaks_indices_2)
    if check_sum == len(rpeaks_indices_1):
        logging.info("两种取值方式结果相同 ... ")
    else:
        logging.info("两种取值方式结果不同，退出 ...")
        sys.exit(1)

    # 与 christov_segmenter 接口一致的还有 hamilton_segmenter
    logging.info("调用接口一致的 hamilton_segmenter 进行R波检测")
    tic = time.time()
    rpeaks = ecg.hamilton_segmenter(signal, sampling_rate=fs)
    toc = time.time()
    logging.info("完成. 用时: %f 秒. " % (toc - tic))
    rpeaks_indices_3 = rpeaks.as_dict()["rpeaks"]
    # 绘波形图和R波位置
    num_plot_samples = 3600
    logging.info("绘制波形图和检测的R波位置 ...")
    sig_plot = signal[:num_plot_samples]
    rpeaks_plot_1 = rpeaks_indices_1[rpeaks_indices_1 <= num_plot_samples]
    plt.figure()
    plt.plot(sig_plot, "g", label="ECG")
    plt.grid(True)
    plt.plot(rpeaks_plot_1, sig_plot[rpeaks_plot_1], "ro", label="christov_segmenter")
    rpeaks_plot_3 = rpeaks_indices_3[rpeaks_indices_3 <= num_plot_samples]
    plt.plot(rpeaks_plot_3, sig_plot[rpeaks_plot_3], "b^", label="hamilton_segmenter")
    plt.legend()
    plt.title(data_path)
    plt.show()
    logging.info("完成.")
    return


if __name__ == '__main__':
    test_rpeaks_simple("./data/ecg_records_117.txt")
    test_rpeaks_simple("./data/ecg_records_103.txt")
    test_rpeaks_simple("./data/ecg_records_119.txt")
