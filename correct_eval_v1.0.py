#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import logging
from biosppy.storage import load_txt
from biosppy.signals import ecg
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

signal_path = "./data/ecg_records_117.txt"
ann_path = "./data/ecg_ann_117.txt"

logging.info("--------------------------------------------------")
signal, _ = load_txt(signal_path)
logging.info("载入信号-%s, 长度 = %d. " % (signal_path, len(signal)))
ann, _ = load_txt(ann_path)
logging.info("载入R峰位置人工标记, 共 %d 个R峰." % (len(ann)))

fs = 360  # 信号采样率 360 Hz
logging.info("调用 hamilton_segmenter 进行R波检测 ...")
tic = time.time()
rpeaks = ecg.hamilton_segmenter(signal, sampling_rate=fs)
toc = time.time()
logging.info("完成. 用时: %f 秒. " % (toc - tic))
rpeaks = rpeaks[0]

logging.info("使用compare_segmentation对比算法结果与人工标记 ...")
tic = time.time()
eval_results = ecg.compare_segmentation(ann, rpeaks, fs, tol=0.02)
toc = time.time()
logging.info("完成. 用时: %f 秒. 返回结果类型为 %s ." % (toc - tic, str(type(eval_results))))

dict_results = eval_results.as_dict()
logging.info("********** 结果报告 *************")
logging.info("* 准确率(acc): %.3f *" % dict_results["acc"])
logging.info("* 总体表现(performance): %.3f *" % dict_results["performance"])
logging.info("*********************************")

correct_tol = 0.05
logging.info("使用correct_rpeaks校正R波位置, 最大校正范围 %.3f 秒 ..." % correct_tol)
tic = time.time()
rpeaks_correct = ecg.correct_rpeaks(signal, rpeaks, fs, tol=correct_tol)
toc = time.time()
logging.info("完成. 用时: %f 秒. 返回结果类型为 %s ." % (toc - tic,
                                             str(type(rpeaks_correct))))
rpeaks_correct = rpeaks_correct.as_dict()["rpeaks"]

logging.info("绘制部分R峰校正前后的位置 ...")
num_plot_samples = 3600
sig_plot = signal[:num_plot_samples]
rpeaks_plot = rpeaks[rpeaks <= num_plot_samples]
rpeaks_correct_plot = rpeaks_correct[rpeaks_correct <= num_plot_samples]
plt.figure()
plt.grid(True)
plt.plot(sig_plot, label="ECG")
plt.plot(rpeaks_plot, sig_plot[rpeaks_plot], "ro", label="rpeaks")
plt.plot(rpeaks_correct_plot, sig_plot[rpeaks_correct_plot], "b*", label="corrected rpeaks")
plt.legend()
plt.show()
logging.info("绘图完成.")

logging.info("使用biosppy.signals.ecg.ecg 综合处理 ...")
tic = time.time()
summary_result = ecg.ecg(sig_plot, fs, True)
toc = time.time()
logging.info("完成. 用时: %f 秒. 返回结果类型为 %s ." % (toc - tic,
                                             str(type(summary_result))))

summary_result = summary_result.as_dict()
