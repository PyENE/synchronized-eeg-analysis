# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

from eega.modwt import MODWT
from eega.sync_eeg import SyncEEG
from eega.config import DATA_PATH
import glob2
import os
import pandas as pd
import re


# DATA_PATH = '../data'

em_xls_file = os.path.join(DATA_PATH, 'em-ry35-unidecode.xlsx')
em_data = pd.read_excel(em_xls_file)
eeg_epochs_paths = glob2.glob(os.path.join(DATA_PATH, 'eeg/**/*/Trig_S1001_XLS/*.mat'))
subject_name = 's06'
text_type = 'M'
eeg_epoch_path = [re.findall('.*/%s/%s/.*' % (subject_name, text_type), eeg_epochs_path)[0]
				  for eeg_epochs_path in eeg_epochs_paths
				  if len(re.findall('.*/%s/%s/.*' % (subject_name, text_type), eeg_epochs_path)) != 0][0]
text_name = 'conflit_israelo_palestinien-m2'
se = SyncEEG(eeg_epoch_path, em_data)
channel_name = se.channel_names[0]
tmin = se.get_first_fixation_time_id(text_name)
tmax = se.get_last_fixation_time_id(text_name)
epoch_id = se.get_epoch_id(text_name)
channel_id = se.get_channel_id(channel_name)
ts = se.mat_data.data[channel_id, :, epoch_id]
phases = se.compute_epoch_phases(text_name, from_zero=True, tmax=se.get_last_fixation_time(text_name))
phases_for_df = sum([sum([[phase[2]]*(phase[1] - phase[0])
							for phase in phases], [])] * 7, [])
wt = MODWT(ts, tmin=tmin, tmax=tmax, margin=-se.mat_data.times[0], nlevels=7, wf='la8')

df = pd.DataFrame(wt.wt)
df = df.reset_index().melt(id_vars=['index'])
df = df.rename(index=str, columns={'index': 'SCALE',
								   'variable': 'TIME',
								   'value': 'VALUE'})
df['TIME'] = df['TIME'].astype('uint16')
df['VALUE'] = df['VALUE'].astype('float16')
df['SCALE'] = df['SCALE'].astype('category')
df['SCALE'] = df['SCALE'].cat.set_categories(range(7))
df['CHANNEL'] = channel_name
df['CHANNEL'] = df['CHANNEL'].astype('category')
df['CHANNEL'] = df['CHANNEL'].cat.set_categories(se.channel_names)
df['TEXT'] = text_name
df['TEXT'] = df['TEXT'].astype('category')
df['TEXT'] = df['TEXT'].cat.set_categories([text_name for text_name in se.text_names
											if len(text_name) != 0])
df['SUBJECT'] = se.subject_name
df['SUBJECT'] = df['SUBJECT'].astype('category')
df['PHASE'] = phases_for_df
df['PHASE'] = df['PHASE'].astype('category')
df['PHASE'] = df['PHASE'].cat.set_categories(range(4))