# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import mne
import os
import scipy.io
from sea import SynchronizedEEGTrial
from sea.config import DATA_PATH
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def synchronized_eeg_trial_init():
    eeg_data = scipy.io.loadmat(os.path.join(DATA_PATH, 's01_sample.mat'),
                                squeeze_me=True, struct_as_record=False)['EEG']
    em_data = pd.read_csv(os.path.join(DATA_PATH, 's01_sample.csv'))
    channel_info = mne.io.read_epochs_eeglab(os.path.join(DATA_PATH, 'synchro_s01_test.set'), verbose='CRITICAL').info
    return SynchronizedEEGTrial(eeg_data, em_data, 's01', 'chasse_oiseaux-f1', channel_info)


def test_synchronized_eeg_new(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.subject_name == 's01'
    assert synchronized_eeg_trial_init.text_name == 'chasse_oiseaux-f1'


def test_get_channel_id(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.get_channel_id('Fp1') == 0


def test_get_time_index(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.get_time_index(0) == 188


def test_get_fixations_event_id(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.em_trial.shape[0] == len(synchronized_eeg_trial_init.get_fixations_event_id())


def test_get_first_fixation_time(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.get_first_fixation_time() == 194


def test_get_last_fixation_time(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.get_last_fixation_time() == 4365


def test_get_first_fixation_time_id(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.get_first_fixation_time_id() == 382


def test_get_last_fixation_time_id(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.get_last_fixation_time_id() == 4553


def test_get_em_epoch_start_time(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.get_em_epoch_start_time() == 194


def test_get_em_epoch_end_time(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.get_em_epoch_end_time() == 4365


def test_is_eeg_epoch_truncated(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.is_eeg_epoch_truncated() == False


def test_get_fixations_time(synchronized_eeg_trial_init):
    assert len(synchronized_eeg_trial_init.get_fixations_time()) == synchronized_eeg_trial_init.em_trial.shape[0]


def test_get_fixed_words(synchronized_eeg_trial_init):
    assert len(synchronized_eeg_trial_init.get_fixed_words()) == synchronized_eeg_trial_init.em_trial.shape[0]


def test_plot_activity(synchronized_eeg_trial_init):
    synchronized_eeg_trial_init.plot_activity('Fp1')


def test_compute_epoch_phases(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.compute_epoch_phases() == [(194, 889, 0), (889, 2213, 1), (2213, 4365, 3)]
    assert synchronized_eeg_trial_init.compute_epoch_phases(from_zero=True) == [(0, 695, 0), (695, 2019, 1), (2019, 4171, 3)]


def test_compute_modwt(synchronized_eeg_trial_init):
    assert synchronized_eeg_trial_init.compute_modwt().shape == (875910, 7)
    assert synchronized_eeg_trial_init.compute_modwt(standardize_trial=0).shape == (875910, 7)
    assert synchronized_eeg_trial_init.compute_modwt(standardize_trial=1).shape == (875910, 7)
    assert synchronized_eeg_trial_init.compute_modwt(standardize_trial=2).shape == (875910, 7)
    assert synchronized_eeg_trial_init.compute_modwt(standardize_trial=3).shape == (875910, 7)
