__author__ = 'Brice Olivier'

from eega import DATA_PATH
from eega import MODWT
from eega import SyncEEG
import glob2
import os
import pandas as pd

em_xls_file = os.path.join(DATA_PATH, 'em-ry35-unidecode.xlsx')
em_data = pd.read_excel(em_xls_file)
eeg_epochs_paths = glob2.glob(os.path.join(DATA_PATH, 'eeg/**/*/Trig_S1001_XLS/*.mat'))

se = SyncEEG(eeg_epochs_paths[0], em_data)
text_name = se.text_names[0]
phases = se.compute_epoch_phases(text_name, from_zero=True)
# phases = [(0, 2399, 0), (2399, 5286, 1)]
for channel_name in se.channel_names:
	tmin = se.get_first_fixation_time_id(text_name)
	tmax = se.get_last_fixation_time_id(text_name)
	ts = se.mat_data.data[se.get_channel_id(channel_name), :,
		 se.get_epoch_id(text_name)]
	wt = MODWT(ts, tmin=tmin, tmax=tmax,
			   margin=150, nlevels=7, wf='la8')
	fixations_time = se.get_fixations_time(text_name, from_zero=True)
	tags = se.get_fixed_words(text_name)
	wt.plot_time_series_and_wavelet_transform_with_phases(phases, 'last5',
														  fixations_time, tags)

