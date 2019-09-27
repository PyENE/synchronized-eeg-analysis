# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sea.config import OUTPUT_PATH
import uuid


PHASE_NAMES = ['Fast Forward', 'Normal Reading', 'Information Search', 'Slow Confirmation']
PHASE_NAMES_SHORT = ['FF', 'NR', 'IS', 'SC']


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

    @staticmethod
    def concat(melted_modwt_dataframes):
        assert all([type(melted_modwt_dataframe) == MeltedMODWTDataFrame
                    for melted_modwt_dataframe in melted_modwt_dataframes])
        melted_modwt_dataframe = pd.concat(melted_modwt_dataframes)
        melted_modwt_dataframe.channel_info = melted_modwt_dataframe[0].channel_info
        return melted_modwt_dataframe

    def plot_var_heatmap(self, last_x_scales=None, robust=False, normalize_power_spectrum=False):
        assert all([col in self.columns for col in ['PHASE', 'CHANNEL', 'SCALE']])
        # nb_phases = self['PHASE'].astype(int).max()
        scale_names = ['sc1', 'sc2', r'$\gamma$ +', r'$\gamma$ -', r'$\beta$', r'$\alpha$', r'$\theta$']
        nb_scales = len(self['SCALE'].unique())
        last_x_scales = nb_scales if (last_x_scales is None) or (last_x_scales > nb_scales) else last_x_scales
        df = self.groupby(['PHASE', 'CHANNEL', 'SCALE']).var().reset_index()
        df = df[df['SCALE'].isin(range(nb_scales - last_x_scales, nb_scales))]
        values = df.VALUE
        if normalize_power_spectrum:
            #values /= (df.SCALE.astype(float) + 1)
            values /= 2**(df.SCALE.astype(float))
        if robust:
            vmin = values.quantile(q=0.10)
            vmax = values.quantile(q=0.90)
        else:
            vmin = values.min()
            vmax = values.max()
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        for i, ax in enumerate(axes.flat):
            if i in df['PHASE'].unique():
                v = df[df['PHASE'] == i].pivot(index='SCALE', columns='CHANNEL', values='VALUE')
                sns.heatmap(v, ax=ax, vmin=vmin, vmax=vmax, cbar=i == 0, cbar_ax=None if i else cbar_ax,
                            yticklabels=scale_names[-last_x_scales:])
                ax.set_title(PHASE_NAMES[i])
                ax.set_xlabel('')
                ax.set_ylabel('')
        # fig.tight_layout()  # seaborn.heatmap ax is tight_layout() incompatible
        plt.show()

    def plot_corr_heatmap(self, last_x_scales=None):
        assert all([col in self.columns for col in ['PHASE', 'CHANNEL', 'SCALE']])
        nb_scales = len(self['SCALE'].unique())
        scale_names = ['sc1', 'sc2', r'$\gamma$ +', r'$\gamma$ -', r'$\beta$', r'$\alpha$', r'$\theta$']
        last_x_scales = nb_scales if (last_x_scales is None) or (last_x_scales > nb_scales) else last_x_scales
        channel_names = self['CHANNEL'].unique().tolist()
        df = self.groupby(['SCALE', 'CHANNEL', 'PHASE'])['VALUE'].apply(lambda x: [elem for elem in x]).reset_index()
        for scale in range(nb_scales - last_x_scales, nb_scales):
            fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
            cbar_ax = fig.add_axes([.91, .3, .03, .4])
            for i, ax in enumerate(axes.flat):
                if i in df.loc[df['SCALE'] == i, 'PHASE'].unique():
                    sub_gb = df.loc[(df.SCALE == scale) & (df.PHASE == i)]
                    corr_mat = np.corrcoef([sub_gb.loc[j, 'VALUE'] for j in sub_gb.index])
                    sns.heatmap(corr_mat, ax=ax, xticklabels=channel_names, yticklabels=channel_names,
                                vmin=0, vmax=1, cbar=i == 0, cbar_ax=None if i else cbar_ax)
                ax.set_title(PHASE_NAMES[i])
                ax.set_xlabel('')
                ax.set_ylabel('')
            fig.suptitle(scale_names[scale])
            plt.show()

    def plot_topomap(self, groupby=None, robust=False, last_x_scales=None,
                     is_file_output=False, normalize_power_spectrum=False):
        self['TEXT_TYPE'] = self['TEXT'].apply(lambda x: x.split('-')[1][0])
        if groupby is not None:
            assert all([col in self.columns for col in groupby])
        nb_scales = len(self['SCALE'].unique()) if 'SCALE' in groupby else 1
        scale_names = ['sc1', 'sc2', r'$\gamma$ +', r'$\gamma$ -', r'$\beta$', r'$\alpha$', r'$\theta$']
        last_x_scales = nb_scales if (last_x_scales is None) or (last_x_scales > nb_scales) else last_x_scales
        nb_phases = self['PHASE'].astype(int).max() + 1 if 'PHASE' in groupby else 1
        subject_names = self['SUBJECT'].unique()
        nb_subjects = len(subject_names) if 'SUBJECT' in groupby else 1
        text_types = self['TEXT_TYPE'].unique()
        nb_text_types = len(text_types) if 'TEXT_TYPE' in groupby else 1
        self['SCALE'] = self['SCALE'].astype(float)
        gb = self[self['SCALE'].isin(range(nb_scales - last_x_scales, nb_scales))].groupby(
            groupby).var().reset_index()
        values = gb.VALUE
        if normalize_power_spectrum:
            #values /= (gb.SCALE.astype(float) + 1)
            values /= 2**(gb.SCALE.astype(float))
        if robust:
            vmin = values.quantile(q=0.10)
            vmax = values.quantile(q=0.90)
        else:
            vmin = values.min()
            vmax = values.max()

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
                    scale_id = nb_scales - 1 - i % (last_x_scales)
                    phase_id = int(i / (last_x_scales))
                    if phase_id in self['PHASE'].unique():
                        gb_values = np.array(self[(self['SCALE'] == scale_id) &
                                                  (self['PHASE'] == phase_id) &
                                                  (self['TEXT_TYPE'].isin(text_type)) &
                                                  (self['SUBJECT'].isin(subject_name))
                                             ].groupby(['CHANNEL']).var().VALUE)
                        if normalize_power_spectrum:
                            #gb_values = gb_values / (scale_id + 1)
                            gb_values = gb_values / (2**scale_id)
                        mne.viz.plot_topomap(gb_values, self.channel_info, axes=ax,
                                             vmin=vmin, vmax=vmax, show=False)
                        if phase_id == self['PHASE'].astype(int).max():
                            ax.set_xlabel(scale_names[scale_id])
                        if scale_id == nb_scales - 1 % (last_x_scales):
                            ax.set_ylabel(PHASE_NAMES_SHORT[phase_id])
                plot_title = ''
                if nb_text_types == 1:
                    plot_title += 'all text types'
                else:
                    plot_title += 'text type %s' % text_type
                if nb_subjects == 1:
                    plot_title += ', all subjects'
                else:
                    plot_title += ', subject %s' % subject_name
                #fig.text(0.5, 0.98, plot_title, ha='center')
                #fig.text(0.5, 0.01, 'scale', ha='center')
                #fig.text(0.01, 0.5, 'phase', va='center', rotation='vertical')
                fig.tight_layout(rect=[0, 0, .9, 1])
                if is_file_output:
                    file_path = uuid.uuid4().hex + '.png'
                    if not os.path.exists(OUTPUT_PATH):
                        os.makedirs(OUTPUT_PATH)
                    file_path = os.path.join(OUTPUT_PATH, file_path)
                    plt.savefig(file_path)
                    print('topomap - %s, saved to %s' % (plot_title, file_path))
                else:
                    mne.viz.utils.plt_show()

