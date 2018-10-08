# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

from eega.config import DATA_PATH
import glob2
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import re


plt.style.use('seaborn')
scales = ['Scale 1', 'Scale 2', 'Gamma h', 'Gamma l', 'Beta', 'Alpha', 'Theta']

# P4 deleted: too high var -- P3
# T7, T8, TP9

channel_names = ['C3', 'C4', 'CP1', 'CP5', 'CP6', 'Cz', 'F3', 'F4', 'F7', 'F8',
				 'FC1', 'FC5', 'FC6', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'Oz',
				 'P7', 'P8', 'PO10', 'PO9', 'Pz']

ree = mne.read_epochs_eeglab('/home/bolivier/cw/ema/eega/data/eeg/s01/A/Trig_S1001_XLS/synchro_s01_test.set')
ree = ree.pick_channels(channel_names)


concat_gb = []
mean_aggregated_file_paths = glob2.glob(os.path.join(DATA_PATH, 'wt-gb/var-*'))
for mean_aggregated_file_path in mean_aggregated_file_paths:
	subject_name = re.findall('\s*s[0-9][0-9]\s*', mean_aggregated_file_path)[0]
	gb = pd.read_csv(mean_aggregated_file_path)
	gb['SUBJECT'] = subject_name
	concat_gb.append(gb)
concat_gb = pd.concat(concat_gb)
concat_gb = concat_gb.loc[concat_gb.CHANNEL.isin(channel_names)]

for mean_aggregated_file_path in mean_aggregated_file_paths:
	subject_name = re.findall('\s*s[0-9][0-9]\s*', mean_aggregated_file_path)[0]
	gb = pd.read_csv(mean_aggregated_file_path)
	gb['SUBJECT'] = subject_name
	gb = gb.loc[gb.CHANNEL.isin(channel_names)]
	ree = ree.reorder_channels(gb.CHANNEL.unique())
	mat = np.array(gb.VALUE).reshape((7, len(channel_names), 4))

	fig, axes = plt.subplots(3, 4)
	for i, ax in enumerate(axes.flat):
		wave_id = int(np.floor(i / 4))
		vmin = concat_gb.loc[concat_gb.SCALE == 6 - wave_id, 'VALUE'].min()
		vmax = concat_gb.loc[concat_gb.SCALE == 6 - wave_id, 'VALUE'].max()
		wave = scales[-1 - wave_id]
		phase = i % 4
		mne.viz.plot_topomap(mat[-1 - wave_id, :, phase], ree.info, axes=ax,
							 vmin=vmin, vmax=vmax, show=False)
		ax.set_title('%s waves - phase %d' % (wave, phase))
	fig.suptitle('%s' % subject_name)
	fig.tight_layout(rect=[0, 0, .9, 1])
	plt.savefig('/home/bolivier/cw/ema/plots/wt-activity-%s.png' % subject_name)
	mne.viz.utils.plt_show()

gb = concat_gb.groupby(['SCALE', 'CHANNEL', 'PHASE'])['VALUE'].mean()
mat = np.array(gb.values).reshape((7, len(channel_names), 4))

fig, axes = plt.subplots(3, 4)
for i, ax in enumerate(axes.flat):
	wave_id = int(np.floor(i / 4))
	vmin = concat_gb.loc[concat_gb.SCALE == 6 - wave_id, 'VALUE'].min()
	vmax = concat_gb.loc[concat_gb.SCALE == 6 -wave_id, 'VALUE'].max()
	#vmin = concat_gb.VALUE.min()
	#vmax = concat_gb.VALUE.max()
	wave = scales[-1 - wave_id]
	phase = i % 4
	mne.viz.plot_topomap(mat[-1 - wave_id, :, phase], ree.info, axes=ax,
						 vmin=vmin, vmax=vmax, show=False)
	wave = scales[int(-1 - np.floor(i / 4))]
	phase = i % 4
	ax.set_title('%s waves - phase %d' % (wave, phase))
fig.suptitle('average subject')
fig.tight_layout(rect=[0, 0, .9, 1])
plt.savefig('/home/bolivier/cw/ema/plots/wt-activity-avg-subject.png')
mne.viz.utils.plt_show()
