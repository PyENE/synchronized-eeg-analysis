# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import os
import scipy.io
from sea import MODWT
from sea.config import DATA_PATH
from .test_synchronized_eeg import synchronized_eeg_init
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


@pytest.mark.usefixtures("synchronized_eeg_init")
def test_plot_modwt_with_phases_and_tags(synchronized_eeg_init):
    fixations_time = synchronized_eeg_init.get_fixations_time(from_zero=True)
    tags = synchronized_eeg_init.get_fixed_words()
    ts = synchronized_eeg_init.eeg_trial[0, :]
    tmin = synchronized_eeg_init.get_first_fixation_time_id()
    tmax = synchronized_eeg_init.get_last_fixation_time_id()
    margin = - synchronized_eeg_init.eeg_times[0]
    phases = synchronized_eeg_init.compute_epoch_phases(from_zero=True,
                                                        tmax=synchronized_eeg_init.get_last_fixation_time())
    wt = MODWT(ts, tmin=tmin, tmax=tmax, margin=margin, nlevels=7, wf='la8')
    wt.plot_time_series_and_wavelet_transform_with_phases(phases, events=fixations_time, tags=tags)