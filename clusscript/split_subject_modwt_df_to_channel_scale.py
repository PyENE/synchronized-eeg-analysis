# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import numpy as np
from optparse import OptionParser
import pandas as pd
import os


parser = OptionParser()
parser.add_option("-i", "--input", dest="input_file_path",
                  help="Path to the subject modwt dataframe file(.csv).",
                  default=None)


if __name__ == "__main__":
	(options, args) = parser.parse_args()
	subject_modwt_df = pd.read_csv(options.input_file_path)
	for scale in subject_modwt_df['SCALE'].unique():
		for channel in subject_modwt_df['CHANNEL'].unique():
			file_path = '../data/wt/sc%s-%s.csv' % (scale, channel)
			sub_df = subject_modwt_df[(subject_modwt_df['SCALE'] == scale)
									  & (subject_modwt_df['CHANNEL'] == channel)]
			if os.path.exists(file_path):
				df = pd.read_csv(file_path)
				sub_df = pd.concat((df, sub_df))
			sub_df.to_csv(file_path, index=False)
