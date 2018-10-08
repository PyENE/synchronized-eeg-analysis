import eega
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pywt


def plot_ts_ws(ts, ws, ylabels):
    tmax = ws.shape[1]
    f, axarr = plt.subplots(ws.shape[0] + 1, sharex=True)
    plt.xlabel('time (ms)')
    for i in range(0, ws.shape[0]):
        axarr[i].plot(range(0, tmax),
                      ws[ws.shape[0] - i - 1, :],
                      linewidth=1)
        axarr[i].set_ylabel("%s" % ylabels[ws.shape[0] - i - 1], rotation=0)
        axarr[i].yaxis.set_label_coords(0.5, 0.8)
        # axarr[i].set_ylim(-0.2, 0.3)
    plt.xlabel('time (ms)')
    axarr[ws.shape[0]].plot(range(0, tmax), ts[0:tmax], linewidth=1)
    axarr[ws.shape[0]].set_ylabel(
        "mean=%.1e, var=%.1e" % (np.mean(ts), np.var(ts)), rotation=0)
    axarr[ws.shape[0]].yaxis.set_label_coords(0.5, 0.8)
    plt.subplots_adjust(wspace=0, hspace=0.2)


gdpdata = pd.read_csv(os.path.join(eega.DATA_PATH, 'GDPcomponents.csv'))

n_levels = 5
modwt_len = gdpdata.shape[0] - gdpdata.shape[0] % 2**n_levels
ts = gdpdata.loc[0:modwt_len - 1, 'govtexp']

modwt_wt = eega.modwt(ts, 'sym8', n_levels)
pywt_wt = pywt.swt(ts, 'sym8', n_levels)
pywt_wt = np.stack(pywt_wt)
pywt_wt = np.vstack((pywt_wt[:, 1, :], pywt_wt[0, 0, :]))
pywt_wt = pywt_wt[np.concatenate((np.arange(4, -1, -1), [5])), :]


ylabels = ['s%d: mean=%.1e, var=%.1e' % (scale + 1, np.mean(modwt_wt[scale, :]),
                                         np.var(modwt_wt[scale, :]))
           for scale in range(0, modwt_wt.shape[0])]
plot_ts_ws(ts, modwt_wt, ylabels)
plt.savefig(os.path.join(eega.PROJECT_PATH, 'test', 'py_modwt_wt.png'))

ylabels = ['s%d: mean=%.1e, var=%.1e' % (scale + 1, np.mean(pywt_wt[scale, :]),
                                         np.var(pywt_wt[scale, :]))
           for scale in range(0, pywt_wt.shape[0])]
plot_ts_ws(ts, pywt_wt, ylabels)
plt.savefig(os.path.join(eega.PROJECT_PATH, 'test', 'py_pywt_wt.png'))

plt.show()
