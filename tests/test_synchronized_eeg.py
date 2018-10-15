# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import scipy.io
from sea import MODWT
from sea import SynchronizedEEG
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def synchronized_eeg_init():
    eeg_data = scipy.io.loadmat('../data/s01_sample.mat', squeeze_me=True, struct_as_record=False)['EEG']
    em_data = pd.read_csv('../data/s01_sample.csv')
    return SynchronizedEEG(eeg_data, em_data, 's01', 'chasse_oiseaux-f1')


def test_synchronized_eeg_new(synchronized_eeg_init):
    assert synchronized_eeg_init.subject_name == 's01'
    assert synchronized_eeg_init.text_name == 'chasse_oiseaux-f1'


def test_get_channel_id(synchronized_eeg_init):
    assert synchronized_eeg_init.get_channel_id('Fp1') == 0


def test_get_time_index(synchronized_eeg_init):
    assert synchronized_eeg_init.get_time_index(0) == 188


def test_get_fixations_event_id(synchronized_eeg_init):
    assert synchronized_eeg_init.em_trial.shape[0] == len(synchronized_eeg_init.get_fixations_event_id())


def test_get_first_fixation_time(synchronized_eeg_init):
    assert synchronized_eeg_init.get_first_fixation_time() == 194


def test_get_last_fixation_time(synchronized_eeg_init):
    assert synchronized_eeg_init.get_last_fixation_time() == 4365


def test_get_first_fixation_time_id(synchronized_eeg_init):
    assert synchronized_eeg_init.get_first_fixation_time_id() == 382


def test_get_last_fixation_time_id(synchronized_eeg_init):
    assert synchronized_eeg_init.get_last_fixation_time_id() == 4553


def test_get_em_epoch_start_time(synchronized_eeg_init):
    assert synchronized_eeg_init.get_em_epoch_start_time() == 194


def test_get_em_epoch_end_time(synchronized_eeg_init):
    assert synchronized_eeg_init.get_em_epoch_end_time() == 4365


def test_is_eeg_epoch_truncated(synchronized_eeg_init):
    assert synchronized_eeg_init.is_eeg_epoch_truncated() == False


def test_get_fixations_time(synchronized_eeg_init):
    assert len(synchronized_eeg_init.get_fixations_time()) == synchronized_eeg_init.em_trial.shape[0]


def test_get_fixed_words(synchronized_eeg_init):
    assert len(synchronized_eeg_init.get_fixed_words()) == synchronized_eeg_init.em_trial.shape[0]


def test_get_baseline_activity(synchronized_eeg_init):
    assert len(synchronized_eeg_init.get_baseline_activity()) == len(synchronized_eeg_init.eeg_channel_names)


def test_plot_baseline_activity(synchronized_eeg_init):
    synchronized_eeg_init.plot_baseline_activity('Fp1')


def test_plot_activity(synchronized_eeg_init):
    synchronized_eeg_init.plot_activity('Fp1')


def test_compute_epoch_phases(synchronized_eeg_init):
    assert synchronized_eeg_init.compute_epoch_phases() == [(194, 889, 0), (889, 2213, 1), (2213, 4365, 3)]
    assert synchronized_eeg_init.compute_epoch_phases(from_zero=True) == [(0, 695, 0), (695, 2019, 1), (2019, 4171, 3)]


def test_compute_modwt(synchronized_eeg_init):
    assert synchronized_eeg_init.compute_modwt().shape == (875910, 7)