# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

from eega.config import DATA_PATH
import glob2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=0.4)
scale_labels = ['Scale 1', 'Scale 2', r'$\gamma^+$', r'$\gamma^-$', r'$\beta$', r'$\alpha$', r'$\theta$']
scale_labels = np.flip(scale_labels, 0)
phase_labels = ['NR', 'SR', 'Un', 'Dec']
channel_names = ['C3', 'C4', 'CP1', 'CP5', 'CP6', 'Cz', 'F3', 'F4', 'F7', 'F8',
				 'FC1', 'FC5', 'FC6', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'Oz',
				 'P7', 'P8', 'PO10', 'PO9', 'Pz','TP10', 'TP9']

concat_gb = []
var_aggregated_file_paths = glob2.glob(os.path.join(DATA_PATH, 'wt-gb/var-*'))
for var_aggregated_file_path in var_aggregated_file_paths:
	subject_name = re.findall('\s*s[0-9][0-9]\s*', var_aggregated_file_path)[0]
	gb = pd.read_csv(var_aggregated_file_path)
	gb['SUBJECT'] = subject_name
	concat_gb.append(gb)
concat_gb = pd.concat(concat_gb)


for subject_name in concat_gb['SUBJECT'].unique():
	fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
	for scale, ax in enumerate(axes.flat):
		d = concat_gb[(concat_gb['SUBJECT'] == subject_name) & (concat_gb['SCALE'] == scale)
		& (concat_gb['CHANNEL'].isin(channel_names))].groupby(
			['CHANNEL', 'PHASE'])['VALUE'].mean().unstack()
		sns.heatmap(d, ax=ax, xticklabels=phase_labels)
		ax.set_title('%s %s' % (subject_name, scale_labels[scale]))
		#ax.set(adjustable='box-forced', aspect='equal')
	plt.show()


fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
for scale, ax in enumerate(axes.flat):
	d = concat_gb[(concat_gb['SCALE'] == scale) & (concat_gb['CHANNEL'].isin(channel_names))].groupby(
		['CHANNEL', 'PHASE'])['VALUE'].mean().unstack()
	sns.heatmap(d, ax=ax, xticklabels=phase_labels)
	ax.set_title('average subject %s' % (scale_labels[scale]))
	#ax.set(adjustable='box-forced', aspect='equal')
plt.show()