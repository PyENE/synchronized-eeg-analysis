# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

from optparse import OptionParser
import os
import pandas as pd


parser = OptionParser()
parser.add_option("-i", "--input", dest="input_file_path",
                  help="Path to the subject modwt dataframe file(.csv).",
                  default=None)
parser.add_option("-f", "--fun", dest="fun",
				  help="groupby function to apply on pandas dataframe (\"mean\" or \"var\")",
				  default=None)


if __name__ == "__main__":
	available_fun = ['mean', 'var']
	(options, args) = parser.parse_args()
	subject_modwt_df = pd.read_csv(options.input_file_path)
	if options.fun in available_fun:
		if options.fun == 'var':
			subject_gb = subject_modwt_df.groupby(['SCALE', 'CHANNEL', 'PHASE'])['VALUE'].var()
		else:
			subject_gb = subject_modwt_df.groupby(['SCALE', 'CHANNEL', 'PHASE'])['VALUE'].mean()
		subject_gb.reset_index().to_csv(os.path.join('..', 'data', 'wt-gb', options.fun + '-' +
													 options.input_file_path.split('/')[-1]))
	else:
		print('--fun not in %s' % available_fun)
