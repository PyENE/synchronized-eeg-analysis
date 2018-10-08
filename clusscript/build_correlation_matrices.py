# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import numpy as np
from optparse import OptionParser
import pandas as pd


parser = OptionParser()
parser.add_option("-i", "--input", dest="input_file_path",
                  help="Path to the subject modwt dataframe file(.csv).",
                  default=None)


if __name__ == "__main__":
	(options, args) = parser.parse_args()
	df = pd.read_csv(options.input_file_path)
	subject_name = options.input_file_path.split('/')[-1].split('.')[0]
	gb = df.groupby(['SCALE', 'CHANNEL', 'PHASE'])['VALUE'].apply(lambda x: [elem for elem in x]).reset_index()
	for scale in df['SCALE'].unique():
		for phase in df['PHASE'].unique():
			sub_gb = gb.loc[(gb.SCALE == scale) & (gb.PHASE == phase)]
			corr_mat = np.corrcoef([sub_gb.loc[i, 'VALUE'] for i in sub_gb.index])
			np.save('../data/wt-gb/%s-%s-%s' % (subject_name, scale, phase), corr_mat)


"""
>>> gbr_s6_p0.CHANNEL.unique()
array(['C3', 'C4', 'CP1', 'CP5', 'CP6', 'Cz', 'F3', 'F4', 'F7', 'F8',
       'FC1', 'FC5', 'FC6', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'Oz', 'P3',
       'P4', 'P7', 'P8', 'PO10', 'PO9', 'Pz', 'T7', 'T8', 'TP10', 'TP9'], dtype=object)
"""