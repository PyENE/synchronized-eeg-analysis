# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

from eega.sync_eeg import SyncEEG
from eega.config import DATA_PATH
from eega.modwt import MODWT
import glob2
import numpy as np
import os
import pandas as pd



channels = ['P7', 'P8', 'Oz', 'Pz', 'Cz', 'Fz', 'F7', 'F8']
scales = ['Scale 1', 'Scale 2', 'Gamma h',
		  'Gamma l', 'Beta', 'Alpha', 'Theta']


em_xls_file = os.path.join(DATA_PATH, 'em-ry35-unidecode.xlsx')
em_data = pd.read_excel(em_xls_file)
eeg_epochs_paths = glob2.glob(os.path.join(DATA_PATH, 'eeg/**/*/Trig_S1001_XLS/*.mat'))
eeg_epochs_path = eeg_epochs_paths[28]  # 28, for mistis presentation
subject_name = eeg_epochs_path.split('/')[8]
text_type = eeg_epochs_path.split('/')[9]
se = SyncEEG(eeg_epochs_path, em_data)
text_name = 'art_contemporain-f1'
text_name = se.text_names[1]
# text_name = se.text_names[6]  # art-contemporain-f1
tmin = se.get_first_fixation_time_id(text_name)
tmax = se.get_last_fixation_time_id(text_name)
epoch_id = se.get_epoch_id(text_name)
phases = se.compute_epoch_phases(text_name, from_zero=True)

for channel in channels:
	channel_id = se.get_channel_id(channel)
	ts = se.mat_data.data[channel_id , :, epoch_id]
	wt = MODWT(ts, tmin=tmin, tmax=tmax,
			   margin=150, nlevels=7, wf='la8')
	fixations_time = se.get_fixations_time(text_name, from_zero=True)
	tags = se.get_fixed_words(text_name)
	wt.plot_time_series_and_wavelet_transform_with_phases(phases, scales='all',
														  events=fixations_time, tags=tags)
	print([np.var(wt.wt[:, phase[0]:phase[1]], 1) for phase in phases])
	print([np.var(wt[phase[0]:phase[1]]) for phase in phases])



