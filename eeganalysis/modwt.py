# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import matplotlib.pyplot as plt
import numpy as np
from rpy2 import robjects
from rpy2.robjects.packages import importr


class MODWT(np.ndarray):
	def __new__(cls, input_array, tmin=0, tmax=None, margin=0, nlevels=2, wf="la8"):
		obj = np.asarray(input_array).view(cls)
		obj.tmin = tmin
		obj.tmax = tmax
		obj.margin = margin
		obj.nlevels = nlevels
		obj.wf = wf
		obj.wt = []
		obj.compute_modwt()
		return obj

	def __array_finalize__(self, obj):
		if obj is None: return
		self.tmin = getattr(obj, 'tmin', 0)
		self.tmax = getattr(obj, 'tmax', None)
		self.margin = getattr(obj, 'margin', 0)
		self.nlevels = getattr(obj, 'nlevels', 2)
		self.wf = getattr(obj, 'wf', "la8")

	def compute_modwt(self):
		importr("waveslim")
		modwt = robjects.r['modwt']
		phase_shift = robjects.r['phase.shift']
		if self.tmax is None:
			self.tmax = len(self)
		left_margin = self.margin
		right_margin = self.margin
		if self.tmin - self.margin < 0:
			left_margin = self.tmin
			print('left_margin exceeded, set to %f' % left_margin)
		if self.tmax + self.margin > len(self):
			right_margin = len(self) - self.tmax
			print('right_margin exceeded, set to %f' % right_margin)
		censor = ((self.tmax + right_margin) - (self.tmin - left_margin)) %\
				 2 ** self.nlevels
		if censor > left_margin + right_margin:
			print('issue')
		left_censor = int(np.ceil(censor / 2))
		right_censor = int(np.floor(censor / 2))
		"""
		print('len ts %d, tmin %d, tmax %d, left_margin %d, right_margin %d,'
			  'left_censor %d, right_censor %d' % (len(self), self.tmin, self.tmax,
												   left_margin, right_margin,
												   left_censor, right_censor))
		"""
		wt = modwt(robjects.FloatVector(
			self[self.tmin - left_margin + left_censor : self.tmax + right_margin - right_censor]),
			wf=self.wf, n_levels=self.nlevels)
		wt = phase_shift(wt, 'la8')
		wt = np.array(wt)
		if type(wt[0]) != np.ndarray:
			return
		wt = wt[:-1, :]
		wt = wt[:, left_margin - left_censor: right_censor - right_margin]
		if len(self[self.tmin:self.tmax]) == wt.shape[1]:
			self.wt = wt

	def plot_time_series_and_wavelet_transform(self):
		if self.wt == []:
			return
		time_series = self[self.tmin:self.tmax]
		f, axarr = plt.subplots(self.wt.shape[0] + 1, sharex=True)
		ylabels = ["s" + str(scale + 1) for scale in range(0, self.wt.shape[0])]
		plt.xlabel('time (ms)')
		for i in range(0, self.wt.shape[0]):
			axarr[i].plot(range(len(time_series)),
						  self.wt[self.wt.shape[0] - i - 1, :],
						  linewidth=1)
			axarr[i].set_ylabel("%s" % ylabels[self.wt.shape[0] - i - 1], rotation=0)
		axarr[self.wt.shape[0]].plot(range(len(time_series)), time_series, linewidth=1)
		axarr[self.wt.shape[0]].set_ylabel('TS', rotation=0)
		plt.subplots_adjust(wspace=0, hspace=0.2)
		plt.show()

	def plot_time_series_and_wavelet_transform_with_phases(self, phases, scales='all', events=None, tags=None):
		assert self.wt != []
		if scales == 'last3':
			s = 3
		elif scales == 'last5':
			s = 5
		else:
			s = self.nlevels
		time_series = self[self.tmin:self.tmax]
		f, axarr = plt.subplots(s + 1, sharex=True)
		ylabels = ["sc" + str(scale + 1) for scale in range(0, self.nlevels)]
		for i, ax in enumerate(axarr[:-1]):
			for phase in phases:
				ax.plot(range(phase[0], phase[1]),
						self.wt[self.nlevels - i - 1, range(phase[0], phase[1])],
						color='C%d' % phase[2], linewidth=1)
			ax.set_ylabel("%s" % ylabels[self.nlevels - i - 1], rotation=0)
		for phase in phases:
			axarr[s].plot(range(phase[0], phase[1]),
						  time_series[range(phase[0], phase[1])],
						  color='C%d' % phase[2], linewidth=1)
		if events is not None:
			for event in events:
				for ax in axarr:
					ax.axvline(x=event, color='black', linewidth=1, linestyle=':')
			if tags is not None:
				for i, tag in enumerate(tags):
					axarr[s].text(x=events[i], y=-0.10, s=tag, size='xx-small', rotation=90)
		axarr[s].set_ylabel('TS', rotation=0)

		plt.subplots_adjust(wspace=0, hspace=0.2)
		plt.tight_layout()
		plt.show()


