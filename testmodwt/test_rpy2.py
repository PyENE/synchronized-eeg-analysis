import matplotlib.pyplot as plt
import numpy as np
from rpy2 import robjects
from rpy2.robjects.packages import importr

utils = importr("waveslim")
rreadcsv = robjects.r['read.csv']
rmodwt = robjects.r['modwt']
rdim = robjects.r['dim']
rphaseshift = robjects.r['phase.shift']

gdpdata = rreadcsv("~/cw/ema/eega/data/GDPcomponents.csv")
ts = gdpdata[2]
n_levels = 5
modwt_len = len(ts) - len(ts) % 2**n_levels
ts = ts[0:modwt_len]

wt = rmodwt(ts, wf="la8", n_levels=5)
s_wt = rphaseshift(wt, 'la8')

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

s_wt = np.array(s_wt)
ylabels = ['s%d: mean=%.1e, var=%.1e' % (scale + 1, np.mean(s_wt[scale, :]),
                                         np.var(s_wt[scale, :]))
           for scale in range(0, s_wt.shape[0])]

plot_ts_ws(ts, s_wt, ylabels)
plt.show()