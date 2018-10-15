# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import scipy.io
from sea import MODWT
import pytest


@pytest.fixture(scope="module")
def modwt_init():
    eeg_data = scipy.io.loadmat('../data/s01_sample.mat', squeeze_me=True, struct_as_record=False)['EEG']
    text_id = 0
    channel_id = 0
    return MODWT(eeg_data.data[channel_id, :, text_id], tmin=400,
                 tmax=2000, margin=100, nlevels=7, wf='la8')


def test_modwt(modwt_init):
    assert modwt_init.wt.shape[0] == 7
    assert modwt_init.wt.shape[1] == 1600


def test_plot_modwt(modwt_init):
    modwt_init.plot_time_series_and_wavelet_transform()
