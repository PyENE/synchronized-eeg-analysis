# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import pytest
from sea import MeltedMODWTDataFrame
from sea import SynchronizedEEG
from .test_synchronized_eeg import synchronized_eeg_init


@pytest.fixture(scope="module")
@pytest.mark.usefixtures("synchronized_eeg_init")
def melted_modwt_dataframe_init(synchronized_eeg_init):
    return synchronized_eeg_init.compute_modwt(standardize_trial=2)


def test_topomap(melted_modwt_dataframe_init):
    melted_modwt_dataframe_init.plot_topomap(groupby=['SCALE', 'PHASE'], robust=True, is_file_output=True)
