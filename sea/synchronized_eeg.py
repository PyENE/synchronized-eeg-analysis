# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

from .config import (SUBJECT_COL, TEXT_COL, FIXATION_LATENCY_COL, FIRST_FIXATION_COL,
                     LAST_FIXATION_COL, FIXED_WORD_COL, HUE_COL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .melted_modwt_dataframe import MeltedMODWTDataFrame
from .modwt import MODWT


class SynchronizedEEG:
    def __new__(cls, eeg_data, em_data, subject_name, text_name, channel_info):
        text_list = [epoch.textname for epoch in eeg_data.epoch]
        em_trial = em_data.loc[(em_data[TEXT_COL] == text_name) &
                                    (em_data[SUBJECT_COL] == subject_name)]
        if text_name not in text_list or em_trial.empty:
            print('Trial not found in data')
            return None
        epoch_id = text_list.index(text_name)
        instance = super().__new__(cls)
        instance.subject_name = subject_name
        instance.text_name = text_name
        instance.em_trial = em_trial
        instance.eeg_trial = eeg_data.data[:, :, epoch_id]
        instance.eeg_times = eeg_data.times
        instance.eeg_epoch = eeg_data.epoch[epoch_id]
        instance.eeg_channel_names = [chanloc.labels for chanloc in eeg_data.chanlocs]
        instance.channel_info = channel_info
        return instance

    def get_channel_id(self, channel_name):
        if channel_name in self.eeg_channel_names:
            return self.eeg_channel_names.index(channel_name)
        return -1

    def get_time_index(self, time):
        return np.where(self.eeg_times == time)[0][0]

    def get_fixations_event_id(self):
        return np.where([type(line) == int for line in self.eeg_epoch.numlinexls])[0]

    def get_first_fixation_time(self):
        first_fixation_id = self.get_fixations_event_id()[0]
        return self.eeg_epoch.eventlatency[first_fixation_id]

    def get_last_fixation_time(self):
        last_fixation_id = self.get_fixations_event_id()[-1]
        return self.eeg_epoch.eventlatency[last_fixation_id]

    def get_first_fixation_time_id(self):
        return np.where(self.eeg_times == self.get_first_fixation_time())[0][0]

    def get_last_fixation_time_id(self):
        return np.where(self.eeg_times == self.get_last_fixation_time())[0][0]

    def get_em_epoch_start_time(self):
        return self.em_trial.loc[self.em_trial[FIRST_FIXATION_COL] == 1, FIXATION_LATENCY_COL].values[0]

    def get_em_epoch_end_time(self):
        return self.em_trial.loc[self.em_trial[LAST_FIXATION_COL] == 1, FIXATION_LATENCY_COL].values[0]

    def is_eeg_epoch_truncated(self):
        return self.get_last_fixation_time() > self.get_em_epoch_end_time()

    def get_fixations_time(self, from_zero=False):
        fixations_latency = np.array(self.em_trial[FIXATION_LATENCY_COL])
        if from_zero:
            return fixations_latency - fixations_latency[0]
        return fixations_latency

    def get_fixed_words(self):
        return np.array(self.em_trial[FIXED_WORD_COL])

    def plot_activity(self, channel_name):
        channel_id = self.get_channel_id(channel_name)
        ts = self.eeg_trial[channel_id, :]
        x = self.eeg_times
        plt.plot(x, ts)
        plt.show()

    def compute_epoch_phases(self, from_zero=False, tmax = None):
        phases = []
        start = self.em_trial.loc[self.em_trial.index[0], FIXATION_LATENCY_COL]
        phase = self.em_trial.loc[self.em_trial.index[0], HUE_COL]
        old_phase = phase
        for i in self.em_trial.index[1:-1]:
            phase = self.em_trial.loc[i, HUE_COL]
            if phase != old_phase:
                end = self.em_trial.loc[i, FIXATION_LATENCY_COL]
                phases.append((start, end, old_phase))
                start = end
                old_phase = phase
        end = self.em_trial.loc[self.em_trial.index[-1], FIXATION_LATENCY_COL]
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
            em_start = self.em_trial.loc[self.em_trial.index[0], FIXATION_LATENCY_COL]
            phases = [(phase[0] - em_start, (phase[1] - em_start), phase[2])
                      for phase in phases]
        return phases

    @staticmethod
    def standardize_time_series(time_series, baseline, method):
        """Time domain or frequency domain, average trial or single trial, additive or gain model"""
        if method == 'gain':
            time_series = ((time_series.T - np.mean(baseline, 1)) / np.std(baseline, 1)).T
        elif method == 'additive':
            time_series = (time_series.T / np.mean(baseline, 1)).T
        else:
            print('SynchronizedEEG.standardize_time_series(method): method unknown')
        return time_series

    def compute_modwt(self, standardize_trial=None, margin='max', nlevels=7, wf='la8'):
        """
        :param standardize_trial: |[0;8]|
        0: single trial, time domain, gain model
        1: single trial, time domain, additive model
        2: single trial, time-frequency domain, gain model
        3: single trial, time-frequency domain, additive model
        4: average trial, time domain, gain model
        5: average trial, time domain, additive model
        6: average trial, time-frequency domain, gain model
        7: average trial, time-frequency domain, additive model
        :param margin: int or 'max'
        :param nlevels: int
        :param wf: string
        wavelet function
        :return: pandas.DataFrame
        """
        if margin == 'max':
            margin = - self.eeg_times[0]
        tmin = self.get_first_fixation_time_id()
        tmax = self.get_last_fixation_time_id()
        phases = self.compute_epoch_phases(from_zero=True, tmax=self.get_last_fixation_time())
        melted_modwt_df = []
        phases_for_df = sum([sum([[phase[2]] * (phase[1] - phase[0]) for phase in phases], [])] * nlevels, [])
        index_zero = self.get_time_index(0)
        if standardize_trial in [0, 1]:
            baseline = self.eeg_trial[:, 0:index_zero]
            if standardize_trial == 0:
                self.eeg_trial = self.standardize_time_series(self.eeg_trial, baseline, method='gain')
            elif standardize_trial == 1:
                self.eeg_trial = self.standardize_time_series(self.eeg_trial, baseline, method='additive')
        for channel_name in self.eeg_channel_names:
            channel_id = self.get_channel_id(channel_name)
            ts = self.eeg_trial[channel_id , :]
            if standardize_trial in [2, 3]:
                wt = MODWT(ts, tmin=self.get_time_index(-100), tmax=tmax,
                           margin=margin, nlevels=7, wf='la8')
                wt_baseline = wt.wt[:, 0:100]
                wt_wt = wt.wt[:, tmin - (-self.eeg_times[0] - 100):]
                if standardize_trial == 2:
                    wt_wt = self.standardize_time_series(wt_wt, wt_baseline, method='gain')
                else:
                    wt_wt = self.standardize_time_series(wt_wt, wt_baseline, method='additive')
                wt.wt = wt_wt
                wt.time_series = ts[tmin:tmax]
            else:
                wt = MODWT(ts, tmin=tmin, tmax=tmax, margin=margin, nlevels=nlevels, wf=wf)
            if len(wt.wt) == 0:
                print('wt truncated and removed for %s - %s - %s' % (
                    self.subject_name, self.text_name, channel_name))
            else:
                """
                fixations_time = self.get_fixations_time(from_zero=True)
                tags = self.get_fixed_words(text_name)
                wt.plot_time_series_and_wavelet_transform_with_phases(phases, scales='last_three',
                                                                      events=fixations_time, tags=tags)
                """
                melted_modwt_df.append(self._wt_to_melted_df(wt, phases_for_df, channel_name))
        return MeltedMODWTDataFrame(pd.concat(melted_modwt_df), channel_info = self.channel_info)

    def _wt_to_melted_df(self, wt, phases, channel_name):
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
        df['CHANNEL'] = df['CHANNEL'].cat.set_categories(self.eeg_channel_names)
        df['TEXT'] = self.text_name
        df['TEXT'] = df['TEXT'].astype('category')
        df['SUBJECT'] = self.subject_name
        df['SUBJECT'] = df['SUBJECT'].astype('category')
        df[HUE_COL] = phases
        df[HUE_COL] = df[HUE_COL].astype('category')
        df[HUE_COL] = df[HUE_COL].cat.set_categories(range(4))
        return df
