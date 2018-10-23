# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

class MeltedMODWTDataFrame(pd.DataFrame):
    """
    TODO:
    * subsample df to select channels (with missing channel interpolation), subjects
    * topomaps per stg
    * corr per stg
    """

    _metadata = ['channel_info']

    def __init__(self, *args, **kwargs):
        channel_info = kwargs.pop('channel_info', None)
        super(MeltedMODWTDataFrame, self).__init__(*args, **kwargs)
        self.channel_info = channel_info

    @property
    def _constructor(self):
        return MeltedMODWTDataFrame

    def plot_topomap(self, groupby=None, robust=False, last_x_scales=None):
        """
        TODO: make sure channel_order is the same after DataFrame.groupby: can be SCALE, PHASE, SUBJECT, TEXT_TYPE
        TODO: make groupby not static
        """
        self['TEXT_TYPE'] = self['TEXT'].apply(lambda x: x.split('-')[1][0])
        if groupby is not None:
            assert all([col in self.columns for col in self.columns])

        if robust:
            vmin = self.groupby(groupby).var().VALUE.quantile(q=0.10)
            vmax = self.groupby(groupby).var().VALUE.quantile(q=0.90)
        else:
            vmin = self.groupby(groupby).var().VALUE.min()
            vmax = self.groupby(groupby).var().VALUE.max()

        nb_scales = len(self['SCALE'].unique()) if 'SCALE' in groupby else 1
        last_x_scales = nb_scales if (last_x_scales is None) or (last_x_scales > nb_scales) else last_x_scales
        nb_phases = self['PHASE'].astype(int).max() if 'PHASE' in groupby else 1
        subject_names = self['SUBJECT'].unique()
        nb_subjects = len(subject_names) if 'SUBJECT' in groupby else 1
        text_types = self['TEXT_TYPE'].unique()
        nb_text_types = len(text_types) if 'TEXT_TYPE' in groupby else 1

        for text_type_id  in range(nb_text_types):
            if nb_text_types == 1:
                text_type = text_types
            else:
                text_type = text_types[text_type_id]
            for subject_id in range(nb_subjects):
                if nb_subjects == 1:
                    subject_name = subject_names
                else:
                    subject_name = subject_names[subject_id]
                fig, axes = plt.subplots(nrows=nb_phases, ncols=last_x_scales, sharex=True, sharey=True)
                for i, ax in enumerate(axes.flat):
                    scale_id = i % (last_x_scales)
                    phase_id = int(i / (last_x_scales))
                    if phase_id in self['PHASE'].unique():
                        gb_values = np.array(self[(self['SCALE'] == scale_id) &
                                                  (self['PHASE'] == phase_id) &
                                                  (self['TEXT_TYPE'].isin(text_type)) &
                                                  (self['SUBJECT'].isin(subject_name))
                                             ].groupby(['CHANNEL']).var().VALUE)
                        mne.viz.plot_topomap(gb_values, self.channel_info, axes=ax,
                                             vmin=vmin, vmax=vmax, show=False)
                plot_title = ''
                if nb_text_types == 1:
                    plot_title += 'all text types'
                else:
                    plot_title += 'text type %s' % text_type
                if nb_subjects == 1:
                    plot_title += ', all subjects'
                else:
                    plot_title += ', subject %s' % subject_name
                fig.text(0.5, 0.98, plot_title, ha='center')
                fig.text(0.5, 0.01, 'scale', ha='center')
                fig.text(0.01, 0.5, 'phase', va='center', rotation='vertical')
                fig.tight_layout()
                mne.viz.utils.plt_show()
