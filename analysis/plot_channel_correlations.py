__author__ = 'Brice Olivier'

from eega.config import DATA_PATH
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=0.6)
subject_name = 's01'
scale_labels = ['Scale 1', 'Scale 2', r'$\gamma^+$', r'$\gamma^-$', r'$\beta$', r'$\alpha$', r'$\theta$']
phase_labels = ['NR', 'SR', 'Un', 'Dec']
channel_names = ['C3', 'C4', 'CP1', 'CP5', 'CP6', 'Cz', 'F3', 'F4', 'F7', 'F8',
				 'FC1', 'FC5', 'FC6', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'Oz', 'P3',
				 'P4', 'P7', 'P8', 'PO10', 'PO9', 'Pz', 'T7', 'T8', 'TP10', 'TP9']


for scale_id in range(4, 7):
	fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
	cbar_ax = fig.add_axes([.91, .3, .03, .4])
	for phase, ax in enumerate(axes.flat):
		file_path = os.path.join(DATA_PATH, 'corrmat', '%s-%s-%s.npy' % (subject_name, scale_id, phase))
		d = np.load(file_path)
		sns.heatmap(d, ax=ax, xticklabels=channel_names, yticklabels=channel_names,
                    vmin=0, vmax=0.8, cbar=phase == 0, cbar_ax=None if phase else cbar_ax)
		ax.set_title('%s %s %s' % (subject_name, scale_labels[scale_id], phase_labels[phase]))
		ax.set(adjustable='box-forced', aspect='equal')
	plt.show()


