# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import matplotlib.pyplot as plt
import numpy as np
from rpy2 import robjects
from rpy2.robjects.packages import importr


importr("waveslim")
modwt = robjects.r['modwt']
phase_shift = robjects.r['phase.shift']


class MODWT():
    def __init__(self, input_array, tmin=0, tmax=None, margin=0, nlevels=2, wf='la8'):
        self.time_series = input_array
        tmax = tmax if (tmax is not None) and (tmax <= len(input_array)) else len(input_array)
        self.nlevels = nlevels
        self.wf = wf
        self.wt = np.array([])
        self._compute_modwt(tmin, tmax, margin)

    def _compute_modwt(self, tmin, tmax, margin):
        left_margin = right_margin = margin
        if tmin - margin < 0:
            left_margin = tmin
        if tmax + margin > len(self.time_series):
            right_margin = len(self.time_series) - tmax
        censor = ((tmax + right_margin) - (tmin - left_margin)) % 2 ** self.nlevels
        left_censor = int(np.ceil(censor / 2))
        right_censor = int(np.floor(censor / 2))
        wt = modwt(robjects.FloatVector(
            self.time_series[tmin - left_margin + left_censor : tmax + right_margin - right_censor]),
            wf=self.wf, n_levels=self.nlevels)
        wt = phase_shift(wt, 'la8')
        wt = np.array(wt)
        if type(wt[0]) == np.ndarray:
            wt = wt[:-1, :]
            wt = wt[:, left_margin - left_censor: right_censor - right_margin]
            if wt.shape[1] == len(self.time_series[tmin:tmax]):
                self.time_series = self.time_series[tmin:tmax]
                self.wt = wt

    def plot_time_series_and_wavelet_transform(self):
        assert self.wt != []
        f, axarr = plt.subplots(self.wt.shape[0] + 1, sharex=True)
        ylabels = ["s" + str(scale + 1) for scale in range(0, self.wt.shape[0])]
        plt.xlabel('time (ms)')
        for i in range(0, self.wt.shape[0]):
            axarr[i].plot(range(len(self.time_series)),
                          self.wt[self.wt.shape[0] - i - 1, :],
                          linewidth=1)
            axarr[i].set_ylabel("%s" % ylabels[self.wt.shape[0] - i - 1], rotation=0)
        axarr[self.wt.shape[0]].plot(range(len(self.time_series)), self.time_series, linewidth=1)
        axarr[self.wt.shape[0]].set_ylabel('TS', rotation=0)
        plt.subplots_adjust(wspace=0, hspace=0.2)
        plt.show()

    def plot_time_series_and_wavelet_transform_with_phases(self, phases, last_x_scales=None, events=None, tags=None):
        assert self.wt != []
        last_x_scales = self.nlevels if (last_x_scales is None) or (last_x_scales > self.nlevels) else last_x_scales
        f, axarr = plt.subplots(last_x_scales + 1, sharex=True)
        ylabels = ["sc" + str(scale + 1) for scale in range(0, self.nlevels)]
        for i, ax in enumerate(axarr[:-1]):
            for phase in phases:
                ax.plot(range(phase[0], phase[1]),
                        self.wt[self.nlevels - i - 1, range(phase[0], phase[1])],
                        color='C%d' % phase[2], linewidth=1)
            ax.set_ylabel("%s" % ylabels[self.nlevels - i - 1], rotation=0)
        for phase in phases:
            axarr[last_x_scales].plot(range(phase[0], phase[1]),
                                      self.time_series[range(phase[0], phase[1])],
                                      color='C%d' % phase[2], linewidth=1)
        if events is not None:
            for event in events:
                for ax in axarr:
                    ax.axvline(x=event, color='black', linewidth=1, linestyle=':')
            if tags is not None:
                for i, tag in enumerate(tags):
                    axarr[last_x_scales].text(x=events[i], y=-0.10, s=tag, size='xx-small', rotation=90)
        axarr[last_x_scales].set_ylabel('TS', rotation=0)

        plt.subplots_adjust(wspace=0, hspace=0.2)
        plt.tight_layout()
        plt.show()


