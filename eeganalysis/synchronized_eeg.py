# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import re
from .modwt import MODWT


class SyncEEG:
    def __init__(self, eeg_mat_file, em_data):
        self._init_attributes(eeg_mat_file, em_data)

    def _init_attributes(self, eeg_mat_file, em_data):
        self.eeg_mat_file = eeg_mat_file
        self.mat_data = scipy.io.loadmat(eeg_mat_file, squeeze_me=True,
                                         struct_as_record=False)['EEG']
        self.em_data = em_data
        self.subject_name = re.findall('\s*/s[0-9][0-9]/\s*', eeg_mat_file)[0].split('/')[1]
        self.text_type = re.findall('\s*/[A-Z]/\s*', eeg_mat_file)[0].split('/')[1]
        self.text_names = [epoch.textname for epoch in self.mat_data.epoch]
        self.channel_names = [chanloc.labels for chanloc in self.mat_data.chanlocs]
        self._remove_corrupted_epochs()

    def _remove_corrupted_epochs(self):
        missing_texts_id = np.where([len(text_name) == 0 for text_name in self.text_names])[0]
        if len(missing_texts_id) > 0:
            self.mat_data.epoch = np.delete(self.mat_data.epoch, missing_texts_id)
            self.mat_data.data = np.delete(self.mat_data.data, missing_texts_id, 2)
        self.text_names = [epoch.textname for epoch in self.mat_data.epoch]

    def get_epoch_id(self, text_name):
        if text_name in self.text_names:
            return self.text_names.index(text_name)
        return -1

    def get_channel_id(self, channel_name):
        if channel_name in self.channel_names:
            return self.channel_names.index(channel_name)
        return -1

    def get_time_index(self, time):
        return np.where(self.mat_data.times == time)[0][0]

    def get_fixations_event_id(self, text_name):
        epoch_id = self.get_epoch_id(text_name)
        return np.where([type(line) == int for line in
                         self.mat_data.epoch[epoch_id].numlinexls])[0]

    def get_first_fixation_time(self, text_name):
        epoch_id = self.get_epoch_id(text_name)
        first_fixation_id = self.get_fixations_event_id(text_name)[0]
        return self.mat_data.epoch[epoch_id].eventlatency[first_fixation_id]

    def get_last_fixation_time(self, text_name):
        epoch_id = self.get_epoch_id(text_name)
        last_fixation_id = self.get_fixations_event_id(text_name)[-1]
        return self.mat_data.epoch[epoch_id].eventlatency[last_fixation_id]

    def get_first_fixation_time_id(self, text_name):
        return np.where(self.mat_data.times == self.get_first_fixation_time(text_name))[0][0]

    def get_last_fixation_time_id(self, text_name):
        return np.where(self.mat_data.times == self.get_last_fixation_time(text_name))[0][0]

    def get_em_trial(self, text_name):
        return self.em_data.loc[(self.em_data['TEXT_UNIDECODE'] == text_name) &
                                (self.em_data['SUBJ_NAME'] == self.subject_name)]

    def is_trial_in_em_data(self, text_name):
        return not self.get_em_trial(text_name).empty

    def get_em_epoch_start_time(self, text_name):
        em_trial = self.get_em_trial(text_name)
        return em_trial.loc[em_trial['ISFIRST'] == 1, 'FIX_LATENCY'].values[0]

    def get_em_epoch_end_time(self, text_name):
        em_trial = self.get_em_trial(text_name)
        return em_trial.loc[em_trial['ISLAST'] == 1, 'FIX_LATENCY'].values[0]

    def is_eeg_epoch_truncated(self, text_name):
        return self.get_last_fixation_time(text_name) >\
               self.get_em_epoch_end_time(text_name)

    def get_fixations_time(self, text_name, from_zero=False):
        trial = self.get_em_trial(text_name)
        fixations_latency = np.array(trial['FIX_LATENCY'])
        if from_zero:
            return fixations_latency - fixations_latency[0]
        return fixations_latency

    def get_fixed_words(self, text_name):
        return np.array(self.get_em_trial(text_name)['FIXED_WORD'])

    def get_baseline_activity(self):
        """Task modulates functional connectivity networks in free viewing,
        Seidkhan H. et al. 2017"""
        t0_idx = np.where(self.mat_data.times == -100)[0][0]
        t1_idx = np.where(self.mat_data.times == 0)[0][0]
        return np.mean(self.mat_data.data[:, t0_idx:t1_idx, :], 1)

    def plot_baseline_activity(self, text_name, channel_name):
        epoch_id = self.get_epoch_id(text_name)
        channel_id = self.get_channel_id(channel_name)
        ts = self.mat_data.data[channel_id, :, epoch_id]
        x = self.mat_data.times
        plt.plot(x, ts)
        baseline_activity = self.get_baseline_activity()
        baseline_activity = baseline_activity[channel_id, epoch_id]
        plt.plot(x, [baseline_activity] * len(x))
        plt.show()

    def plot_activity(self, text_name, channel_name):
        epoch_id = self.get_epoch_id(text_name)
        channel_id = self.get_channel_id(channel_name)
        ts = self.mat_data.data[channel_id, :, epoch_id]
        x = self.mat_data.times
        plt.plot(x, ts)
        plt.show()

    def remove_baseline_activity(self):
        data = self.mat_data.data.copy()
        for text_name in self.text_names:
            epoch_id = self.get_epoch_id(text_name)
            baseline_activity = self.get_baseline_activity()
            data[:, :, epoch_id] = (self.mat_data.data[:, :, epoch_id].T -
                                 baseline_activity[:, epoch_id]).T
        return data

    def compute_modwt(self, remove_baseline_activity=True,
                      margin=150, nlevels=7, wf="la8"):
        modwt_df = []
        if margin == 'max':
            margin = - self.mat_data.times[0]
        if remove_baseline_activity:
            self.mat_data.data = self.remove_baseline_activity()
        for text_name in self.text_names:
            print('computing WT for %s - %s...' % (self.subject_name, text_name))
            if not self.is_trial_in_em_data(text_name):
                print('text %s for subj %s not in EM data' %
                      (text_name, self.subject_name))
            else:
                tmin = self.get_first_fixation_time_id(text_name)
                tmax = self.get_last_fixation_time_id(text_name)
                epoch_id = self.get_epoch_id(text_name)
                phases = self.compute_epoch_phases(text_name, from_zero=True,
                                                   tmax=self.get_last_fixation_time(text_name))
                phases_for_df = sum([sum([[phase[2]]*(phase[1] - phase[0])
                                            for phase in phases], [])] * nlevels, [])
                for channel_name in self.channel_names:
                    channel_id = self.get_channel_id(channel_name)
                    ts = self.mat_data.data[channel_id , :, epoch_id]
                    wt = MODWT(ts, tmin=tmin, tmax=tmax,
                               margin=margin, nlevels=nlevels, wf=wf)
                    if len(wt.wt) == 0:
                        print('wt truncated and removed for %s - %s - %s' % (
                        self.subject_name, text_name, channel_name))
                    else:
                        """
                        fixations_time = self.get_fixations_time(text_name, from_zero=True)
                        tags = self.get_fixed_words(text_name)
                        wt.plot_time_series_and_wavelet_transform_with_phases(phases, scales='last_three',
                                                                              events=fixations_time, tags=tags)
                        """
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
                        df['CHANNEL'] = df['CHANNEL'].cat.set_categories(self.channel_names)
                        df['TEXT'] = text_name
                        df['TEXT'] = df['TEXT'].astype('category')
                        df['TEXT'] = df['TEXT'].cat.set_categories([text_name for text_name in self.text_names
                                                                    if len(text_name) != 0])
                        df['SUBJECT'] = self.subject_name
                        df['SUBJECT'] = df['SUBJECT'].astype('category')
                        df['PHASE'] = phases_for_df
                        df['PHASE'] = df['PHASE'].astype('category')
                        df['PHASE'] = df['PHASE'].cat.set_categories(range(4))
                        modwt_df.append(df)
        return pd.concat(modwt_df)

    def compute_epoch_phases(self, text_name, from_zero=False, tmax = None):
        trial = self.get_em_trial(text_name)
        phases = []
        start = trial.loc[trial.index[0], 'FIX_LATENCY']
        phase = trial.loc[trial.index[0], 'PHASE']
        old_phase = phase
        for i in trial.index[1:-1]:
            phase = trial.loc[i, 'PHASE']
            if phase != old_phase:
                end = trial.loc[i, 'FIX_LATENCY']
                phases.append((start, end, old_phase))
                start = end
                old_phase = phase
        end = trial.loc[trial.index[-1], 'FIX_LATENCY']
        phases.append((start, end, phase))
        if tmax is not None:
            phases_tmp = []
            for phase in phases:
                if phase[1] < tmax:
                    phases_tmp.append(phase)
                else:
                    phases_tmp.append((phase[0], tmax, phase[2]))
                    break
            phases = phases_tmp
        if from_zero:
            em_start = trial.loc[trial.index[0], 'FIX_LATENCY']
            phases = [(phase[0] - em_start, (phase[1] - em_start), phase[2])
                      for phase in phases]
        return phases
