# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import os


PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_PATH = os.path.join(PROJECT_PATH, 'sample', 'data')
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'output')

SUBJECT_COL = 'SUBJ_NAME'
TEXT_COL = 'TEXT'
FIXATION_LATENCY_COL = 'FIX_LATENCY'
FIRST_FIXATION_COL = 'ISFIRST'
LAST_FIXATION_COL = 'ISLAST'
FIXED_WORD_COL = 'FIXED_WORD'
HUE_COL = 'PHASE'