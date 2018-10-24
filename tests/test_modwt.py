# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import os
import scipy.io
from sea import MODWT
from sea import DATA_PATH
from .test_synchronized_eeg_trial import synchronized_eeg_trial_init
from sea import SynchronizedEEGTrial
import pytest


@pytest.fixture(scope="module")
def modwt_init():
    eeg_data = scipy.io.loadmat(os.path.join(DATA_PATH, 's01_sample.mat'),
                                squeeze_me=True, struct_as_record=False)['EEG']
    text_id = 0
    channel_id = 0
    return MODWT(eeg_data.data[channel_id, :, text_id], tmin=400,
                 tmax=2000, margin=100, nlevels=7, wf='la8')


def test_modwt(modwt_init):
    assert modwt_init.wt.shape[0] == 7
    assert modwt_init.wt.shape[1] == 1600


def test_plot_modwt(modwt_init):
    modwt_init.plot_time_series_and_wavelet_transform()


@pytest.mark.usefixtures("synchronized_eeg_trial_init")
def test_plot_modwt_with_phases_and_tags(synchronized_eeg_trial_init):
    fixations_time = synchronized_eeg_trial_init.get_fixations_time(from_zero=True)
    tags = synchronized_eeg_trial_init.get_fixed_words()
    ts = synchronized_eeg_trial_init.eeg_trial[0, :]
    tmin = synchronized_eeg_trial_init.get_first_fixation_time_id()
    tmax = synchronized_eeg_trial_init.get_last_fixation_time_id()
    margin = - synchronized_eeg_trial_init.eeg_times[0]
    phases = synchronized_eeg_trial_init.compute_epoch_phases(from_zero=True,
                                                              tmax=synchronized_eeg_trial_init.get_last_fixation_time())
    wt = MODWT(ts, tmin=tmin, tmax=tmax, margin=margin, nlevels=7, wf='la8')
    wt.plot_time_series_and_wavelet_transform_with_phases(phases, events=fixations_time, tags=tags)


@pytest.mark.usefixtures("synchronized_eeg_trial_init")
def test_standardize_eeg(synchronized_eeg_trial_init):
    fixations_time = synchronized_eeg_trial_init.get_fixations_time(from_zero=True)
    ts = synchronized_eeg_trial_init.eeg_trial[0, :]
    tmin = synchronized_eeg_trial_init.get_first_fixation_time_id()
    tmax = synchronized_eeg_trial_init.get_last_fixation_time_id()
    margin = - synchronized_eeg_trial_init.eeg_times[0]

    wt = MODWT(ts, tmin=tmin, tmax=tmax, margin=margin, nlevels=7, wf='la8')
    wt.plot_time_series_and_wavelet_transform(last_x_scales=3, events=fixations_time)

    index_zero = synchronized_eeg_trial_init.get_time_index(0)
    baseline = synchronized_eeg_trial_init.eeg_trial[:, 0:index_zero]

    eeg_trial_additive = SynchronizedEEGTrial.standardize_time_series(synchronized_eeg_trial_init.eeg_trial,
                                                                      baseline, method='additive')
    ts = eeg_trial_additive[0, :]
    wt = MODWT(ts, tmin=tmin, tmax=tmax, margin=margin, nlevels=7, wf='la8')
    wt.plot_time_series_and_wavelet_transform(last_x_scales=3, events=fixations_time)

    eeg_trial_gain = SynchronizedEEGTrial.standardize_time_series(synchronized_eeg_trial_init.eeg_trial,
                                                                  baseline, method='gain')
    ts = eeg_trial_gain[0, :]
    wt = MODWT(ts, tmin=tmin, tmax=tmax, margin=margin, nlevels=7, wf='la8')
    wt.plot_time_series_and_wavelet_transform(last_x_scales=3, events=fixations_time)

    """
    wt = MODWT(ts, tmin=tmin, tmax=tmax, margin=margin, nlevels=7, wf='la8')
    eeg_trial_freq_gain = synchronized_eeg_init.eeg_trial[0, :]
    baseline_wt = MODWT(eeg_trial_freq_gain, tmin=10, tmax=index_zero, margin=8, nlevels=7, wf='la8')
    print(wt.wt.shape, baseline_wt.wt.shape)
    wt.wt = SynchronizedEEGTrial.standardize_time_series(wt.wt, baseline_wt.wt, method='gain')
    wt.plot_time_series_and_wavelet_transform(last_x_scales=3, events=fixations_time)
    """

    wt = MODWT(ts, tmin=synchronized_eeg_trial_init.get_time_index(-100), tmax=tmax,
               margin=margin, nlevels=7, wf='la8')
    wt_baseline = wt.wt[:, 0:100]
    wt_wt = wt.wt[:, tmin-(-synchronized_eeg_trial_init.eeg_times[0] - 100):]
    wt_wt = SynchronizedEEGTrial.standardize_time_series(wt_wt, wt_baseline, method='additive')
    wt.wt = wt_wt
    wt.time_series = ts[tmin:tmax]
    wt.plot_time_series_and_wavelet_transform(last_x_scales=3, events=fixations_time)

    wt = MODWT(ts, tmin=synchronized_eeg_trial_init.get_time_index(-100), tmax=tmax,
               margin=margin, nlevels=7, wf='la8')
    wt_baseline = wt.wt[:, 0:100]
    wt_wt = wt.wt[:, tmin - (-synchronized_eeg_trial_init.eeg_times[0] - 100):]
    wt_wt = SynchronizedEEGTrial.standardize_time_series(wt_wt, wt_baseline, method='gain')
    wt.wt = wt_wt
    wt.time_series = ts[tmin:tmax]
    wt.plot_time_series_and_wavelet_transform(last_x_scales=3, events=fixations_time)
