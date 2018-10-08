# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

from eega import DATA_PATH
from eega import MODWT
from eega import SyncEEG
import glob2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import scipy.fftpack
import scipy.signal as sg


em_xls_file = os.path.join(DATA_PATH, 'em-ry35-unidecode.xlsx')
em_data = pd.read_excel(em_xls_file)
eeg_epochs_paths = glob2.glob(os.path.join(DATA_PATH, 'eeg/**/*/Trig_S1001_XLS/*.mat'))

robust_heatmap=False

se = SyncEEG(eeg_epochs_paths[0], em_data)
#sns.heatmap(np.mean(se.mat_data.data, 1), robust=robust_heatmap, yticklabels=se.channel_names)
#plt.show()
sns.heatmap(se.get_baseline_activity(), robust=robust_heatmap, yticklabels=se.channel_names)
plt.show()
#sns.heatmap(np.mean(se.remove_baseline_activity(), 1), robust=robust_heatmap, yticklabels=se.channel_names)
#plt.show()

eeg_mean_per_channels_per_texts = []
for text_name in se.text_names:
	epoch_id = se.get_epoch_id(text_name)
	first_fixation_id = se.get_first_fixation_time_id(text_name)
	last_fixation_id = se.get_last_fixation_time_id(text_name)
	text_eeg = se.mat_data.data[:, first_fixation_id:last_fixation_id, epoch_id]
	eeg_mean_per_channels_per_texts.append(np.mean(text_eeg, 1))
eeg_mean_per_channels_per_texts = np.column_stack(eeg_mean_per_channels_per_texts)
sns.heatmap(eeg_mean_per_channels_per_texts, robust=robust_heatmap, yticklabels=se.channel_names)
plt.show()

se.mat_data.data = se.remove_baseline_activity()
eeg_mean_per_channels_per_texts = []
for text_name in se.text_names:
	epoch_id = se.get_epoch_id(text_name)
	first_fixation_id = se.get_first_fixation_time_id(text_name)
	last_fixation_id = se.get_last_fixation_time_id(text_name)
	text_eeg = se.mat_data.data[:, first_fixation_id:last_fixation_id, epoch_id]
	eeg_mean_per_channels_per_texts.append(np.mean(text_eeg, 1))
eeg_mean_per_channels_per_texts = np.column_stack(eeg_mean_per_channels_per_texts)
sns.heatmap(eeg_mean_per_channels_per_texts, robust=robust_heatmap, yticklabels=se.channel_names)
plt.show()


"""Var and corr analysis"""

eeg_var_per_channels_per_texts = []
for text_name in se.text_names:
	epoch_id = se.get_epoch_id(text_name)
	first_fixation_id = se.get_first_fixation_time_id(text_name)
	last_fixation_id = se.get_last_fixation_time_id(text_name)
	text_eeg = se.mat_data.data[:, first_fixation_id:last_fixation_id, epoch_id]
	eeg_var_per_channels_per_texts.append(np.var(text_eeg, 1))
eeg_var_per_channels_per_texts = np.column_stack(eeg_var_per_channels_per_texts)
sns.heatmap(eeg_var_per_channels_per_texts, robust=robust_heatmap, yticklabels=se.channel_names)
plt.show()

eeg_corr_per_channel = []
for text_name in se.text_names:
	epoch_id = se.get_epoch_id(text_name)
	first_fixation_id = se.get_first_fixation_time_id(text_name)
	last_fixation_id = se.get_last_fixation_time_id(text_name)
	text_eeg = se.mat_data.data[:, first_fixation_id:last_fixation_id, epoch_id]
	eeg_corr_per_channel.append(text_eeg)
eeg_corr_per_channel = np.concatenate(eeg_corr_per_channel, axis=1)
with sns.axes_style("white"):
	ax = sns.heatmap(np.corrcoef(eeg_corr_per_channel), square=True, yticklabels=se.channel_names,
			xticklabels=se.channel_names, robust=True)
plt.show()


pca = PCA(n_components=10)
pca.fit(eeg_corr_per_channel)
print(pca.explained_variance_ratio_)
pca_data = pca.transform(eeg_corr_per_channel)
fig, ax = plt.subplots()
plt.scatter(pca_data[:, 0], pca_data[:, 1])
for i, txt in enumerate(se.channel_names):
	ax.annotate(txt, (pca_data[i, 0], pca_data[i, 1]))
plt.show()


"""FFT"""

eeg_per_chan = []
for text_name in se.text_names:
	epoch_id = se.get_epoch_id(text_name)
	first_fixation_id = se.get_first_fixation_time_id(text_name)
	last_fixation_id = se.get_last_fixation_time_id(text_name)
	text_eeg = se.mat_data.data[:, first_fixation_id:last_fixation_id, epoch_id]
	eeg_per_chan.append(text_eeg)
eeg_per_chan = np.column_stack(eeg_per_chan)


data = text_eeg[0, :]
plt.plot(data)
plt.show()
N  = 3    # Filter order
Wn = 1./80
B, A = sg.butter(N, Wn, output='ba')
# Second, apply the filter
lpdata = sg.filtfilt(B, A, data)
plt.plot(lpdata)
plt.show()


N = len(data)
T = 1.0 / 1000
x = np.linspace(0.0, N*T, N)
yf = scipy.fftpack.fft(lpdata)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
fig, ax = plt.subplots()
ax.plot(xf[:50], (2.0/N * np.abs(yf[:N//2]))[:50])
plt.show()


""" Baseline removal effects on WT """

text_name = se.text_names[0]
channel_name = se.channel_names[0]

se = SyncEEG(eeg_epochs_paths[0], em_data)
se_without_baseline = SyncEEG(eeg_epochs_paths[0], em_data)
se_without_baseline.mat_data.data = se.remove_baseline_activity()

tmin = se.get_first_fixation_time_id(text_name)
tmax = se.get_last_fixation_time_id(text_name)
ts = se.mat_data.data[se.get_channel_id(channel_name), :,
	 se.get_epoch_id(text_name)]
wt = MODWT(ts, tmin=tmin, tmax=tmax,
		   margin=150, nlevels=7, wf='la8')
wt.plot_time_series_and_wavelet_transform()

ts_without_baseline = se_without_baseline.mat_data.data[
					  se.get_channel_id(channel_name), :,
	 se.get_epoch_id(text_name)]
wt_without_baseline = MODWT(ts_without_baseline, tmin=tmin, tmax=tmax,
							margin=150, nlevels=7, wf='la8')
wt_without_baseline.plot_time_series_and_wavelet_transform()
