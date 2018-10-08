# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

from eega.sync_eeg import SyncEEG
from eega.config import DATA_PATH
import glob2
import os
import pandas as pd
import re


DATA_PATH = '../data'

em_xls_file = os.path.join(DATA_PATH, 'em-ry35-unidecode.xlsx')
em_data = pd.read_excel(em_xls_file)
eeg_epochs_paths = glob2.glob(os.path.join(DATA_PATH, 'eeg/**/*/Trig_S1001_XLS/*.mat'))
# eeg_epochs_paths = eeg_epochs_paths [6:]
# eeg_epochs_paths = [os.path.join(DATA_PATH, 'eeg/s18/A/Trig_S1001_XLS/synchro_s18_test.mat')]
subject_modwt_df = []
subject_name = re.findall('\s*/s[0-9][0-9]/\s*', eeg_epochs_paths[0])[0].split('/')[1]
for eeg_mat_file in eeg_epochs_paths:
	if re.findall('\s*/s[0-9][0-9]/\s*', eeg_mat_file)[0].split('/')[1] != subject_name:
		output_file_name = os.path.join(DATA_PATH, 'wt/%s.csv' % subject_name)
		subject_modwt_df = pd.concat(subject_modwt_df)
		subject_modwt_df.to_csv(output_file_name, float_format='%.3g', index=False)
		print('WT data for %s saved into %s' % (subject_name,output_file_name))
		subject_modwt_df = []
	subject_name = re.findall('\s*/s[0-9][0-9]/\s*', eeg_mat_file)[0].split('/')[1]
	text_type = re.findall('\s*/[A-Z]/\s*', eeg_epochs_paths[0])[0].split('/')[1]
	se = SyncEEG(eeg_mat_file, em_data)
	modwt_df = se.compute_modwt(remove_baseline_activity=False, margin='max')
	subject_modwt_df.append(modwt_df)
