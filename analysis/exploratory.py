# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import colorsys
import glob2
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects.packages import importr
import scipy.io
import seaborn as sns


print('toto')

#%%
utils = importr("waveslim")
rmodwt = robjects.r['modwt']
rphaseshift = robjects.r['phase.shift']


def get_epoch_end_time(epoch, tmax):
    if is_epoch_truncated(epoch):
        return tmax
    return epoch.eventlatency[np.where(epoch.eventtype == 'S1007')[0][0]]


def is_epoch_truncated(epoch):
    return np.sum(epoch.eventtype == 'S1007') == 0


def get_baseline_activity(mat_data):
    """Task modulates functional connectivity networks in free viewing,
    Seidkhan H. et al. 2017"""
    t0_idx = np.where(mat_data.times == -100)[0][0]
    t1_idx = np.where(mat_data.times == 0)[0][0]
    return np.mean(mat_data.data[:, t0_idx:t1_idx, :], 1)


def truncate_epochs(mat_data, epsilon, remove_baseline_activity=True,
                    modwt_ready=True, nlevels=2):
    """truncate epochs in mat_data file

    :param mat_data: .mat data structure
    :param epsilon: data to add before and after trial
    :param remove_baseline_activity: removes baseline activity if set to True
    :param modwt_ready: makes trial size a multiple of 2**nlevels
    :param nlevels: nlevels for modwt
    :return: (truncated data, [(left censor trial 1, right censor trial 1), ..., ()])
    """
    if epsilon < mat_data.times[0]:
        epsilon = - mat_data.times[0]
        print("epsilon set to min value %d" % epsilon)
    if remove_baseline_activity:
        baseline_activity = get_baseline_activity(mat_data)
    censors = np.zeros((mat_data.data.shape[2], 2), dtype=int)
    data = []
    i = 0
    for epoch in mat_data.epoch:
        t0_idx = np.where(
            mat_data.times == (epoch.eventlatency[
                                   epoch.eventtype == 'S1001'][0] - epsilon))[0][0]
        t1_idx = mat_data.times[-1]
        if not is_epoch_truncated(epoch):
            t1_idx = np.where(
                mat_data.times == (epoch.eventlatency[
                                       epoch.eventtype == 'S1007'][0] - epsilon))[0][0]
        if modwt_ready:
            censor = (t1_idx - t0_idx) % 2**nlevels
            left_censor = int(np.floor(censor / 2))
            right_censor = int(np.ceil(censor / 2))
            t0_idx += left_censor
            t1_idx -= right_censor
            censors[i, 0] = left_censor
            censors[i, 1] = right_censor
        if remove_baseline_activity:
            data.append(mat_data.data[:, t0_idx:t1_idx, i].T - baseline_activity[:, 0])
        else:
            data.append(mat_data.data[:, t0_idx:t1_idx, i].T)
        i += 1
    return (np.array(data), censors)


def average_trials(mat_data):
    trunc_data, _ = truncate_epochs(mat_data, 100, remove_baseline_activity=True,
                                    modwt_ready=True, nlevels=7)
    longest_trial = np.max([trial.shape[0] for trial in trunc_data])
    avg_data = np.zeros((longest_trial, mat_data.data.shape[0]))
    dividers = np.zeros(longest_trial)
    for trial in trunc_data:
        avg_data[0:trial.shape[0],:] += trial
        dividers[0:trial.shape[0]] += [1] * trial.shape[0]
    return avg_data.T / dividers


def load_rgb_colors(n_colors=4):
    hues = np.linspace(0, 1, n_colors + 1)[:-1]
    color_list = [(h_i, 0.60, 0.65) for h_i in hues]
    rgb_colors = []
    for color in color_list:
        rgb_colors.append(colorsys.hsv_to_rgb(color[0], color[1], color[2]))
    return rgb_colors


def plot_ts_ws(ts, ws, phases=None):
    rgb_colors = load_rgb_colors(n_colors=4)
    tmax = ws.shape[1]
    f, axarr = plt.subplots(ws.shape[0] + 1, sharex=True)
    ylabels = ["s" + str(scale + 1)
               for scale in range(0, ws.shape[0])]
    plt.xlabel('time (ms)')
    for i in range(0, ws.shape[0]):
        if phases is None:
            axarr[i].plot(range(0, tmax),
                          ws[ws.shape[0] - i - 1, :],
                          linewidth=1)
            axarr[i].set_ylabel("%s" % ylabels[ws.shape[0] - i - 1], rotation=0)
        else:
            for phase in phases:
                axarr[i].plot(range(phase[0], phase[1]),
                              ws[ws.shape[0] - i - 1, phase[0]:phase[1]],
                              color=rgb_colors[phase[2]],
                              linewidth=1)
                axarr[i].set_ylabel("%s" % ylabels[ws.shape[0] - i - 1], rotation=0)
    if phases is None:
        axarr[ws.shape[0]].plot(range(0, tmax), ts[0:tmax], linewidth=1)
    else:
        for phase in phases:
            axarr[ws.shape[0]].plot(range(phase[0], phase[1]),
                                    ts[phase[0]:phase[1]], linewidth=1)
    axarr[ws.shape[0]].set_ylabel('TS', rotation=0)
    plt.subplots_adjust(wspace=0, hspace=0.2)


def get_trial_time(trial):
    return (trial.loc[trial['ISLAST'] == 1, 'REL_FIX_TIME'].values[0] -
            trial.loc[trial['ISFIRST'] == 1, 'REL_FIX_TIME'].values[0])


def trial_to_phases(trial, tmax):
    """@TODO: sync EEG on EM times"""
    phase_sequence = []
    for i in trial.index:
        if trial.loc[i, 'ISLAST']:
            t = tmax - (trial.loc[i, 'REL_FIX_TIME'] - trial.loc[trial.index[0], 'REL_FIX_TIME'])
        else:
            # t = trial.loc[i + 1, 'FIX_LATENCY'] - trial.loc[i, 'FIX_LATENCY']
            t = trial.loc[i + 1, 'REL_FIX_TIME'] - trial.loc[i, 'REL_FIX_TIME']
        state = 0
        if trial.loc[i, 'STATES'] == 2:
            state = 1
        elif trial.loc[i, 'STATES'] == 3:
            state = 2
        elif trial.loc[i, 'STATES'] == 4:
            state = 3
        phase_sequence.append([state] * t)
        # phase_sequence.append([trial.loc[i, 'PHASE']] * t)
    phase_sequence = sum(phase_sequence, [])
    phases = []
    start = 0
    state = phase_sequence[0]
    for i in range(1, len(phase_sequence)):
        if (state != phase_sequence[i]) | (i == len(phase_sequence) - 1):
            end = i
            phases.append((start, end, state))
            start = i - 1
        state = phase_sequence[i]
    return phases


def plot_wt_var_per_chan_per_phase(wt_per_phase_per_channel):
    """per text type, per subject within eeg_data"""
    channels = ['Oz', 'Pz', 'Cz', 'Fz', 'F7', 'F8']
    scales = ['Scale 1', 'Scale 2', 'Gamma h',
              'Gamma l', 'Beta', 'Alpha', 'Theta']
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, axn = plt.subplots(2, 2, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for i, ax in enumerate(axn.flat):
        v = np.var(wt_per_phase_per_channel[i], 1)
        sns.heatmap(np.flip(v, 0), ax=ax,
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax,
                    yticklabels=scales, xticklabels=channels, cmap=cmap)
        ax.set_title('Strategy ' + str(i))
    # fig.tight_layout(rect=[0, 0, .9, 1])
    plt.show()


def plot_wt_cor_per_chan_per_phase(wt_per_phase_per_channel, scale):
    """per text type, per subject within eeg_data"""
    channels = ['Oz', 'Pz', 'Cz', 'Fz', 'F7', 'F8']
    xchannels = channels[:]
    xchannels[-1] = ''
    ychannels = channels[:]
    ychannels[0] = ''
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, axn = plt.subplots(2, 2, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for i, ax in enumerate(axn.flat):
        c = np.corrcoef(wt_per_phase_per_channel[i][scale, :, :].T)
        mask = np.zeros_like(c, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(c, ax=ax, mask=mask, center=0,  vmax=0.8, vmin=0,
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax,
                    yticklabels=ychannels, xticklabels=xchannels, cmap=cmap)
        ax.set_title('Strategy ' + str(i))
    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.show()


#%%
mat_data = scipy.io.loadmat('/home/bolivier/cw/ema/eega/data/eeg/s01/A/Trig_S1001_XLS/synchro_s01_test.mat',
                            squeeze_me=True, struct_as_record=False)['EEG']
eeg_epochs_paths = glob2.glob('/home/bolivier/cw/ema/eega/data/eeg/**/F/Trig_S1001_XLS/*.mat')
plt.style.use('seaborn')
chanlocs = [[chanloc.X, chanloc.Y] for chanloc in mat_data.chanlocs]
chanlocs = np.stack(chanlocs)
channames = [channame.labels for channame in mat_data.chanlocs]
channames = np.stack(channames)

#%% topomap: Trial-averaged EEG mean per subject
fig, axes = plt.subplots(3, 5)
for i, ax in enumerate(axes.flat):
    subject_name = eeg_epochs_paths[i].split("/")[8]
    mat_data = scipy.io.loadmat(eeg_epochs_paths[i], squeeze_me=True,
                                struct_as_record=False)['EEG']
    m = np.mean(average_trials(mat_data), 1)
    mne.viz.plot_topomap(m, chanlocs, axes=ax, show=False, names=channames)
    ax.set_title('%s - %s' % (subject_name, 'F'))
fig.suptitle('Trial-averaged EEG mean per subject')
fig.tight_layout()
mne.viz.utils.plt_show()

#%% topomap: Trial&Subject-averaged EEG mean
activity_per_subject = []
for eeg_epochs_path in eeg_epochs_paths:
    mat_data = scipy.io.loadmat(eeg_epochs_path, squeeze_me=True,
                                struct_as_record=False)['EEG']
    activity_per_subject.append(np.mean(average_trials(mat_data), 1))
mne.viz.plot_topomap(np.squeeze(np.asarray(np.mean(np.matrix(activity_per_subject), 0))),
                     chanlocs, names=channames)
mne.viz.utils.plt_show()

# %% heatmap: Mean per chan per subj per trial
channels = ['Oz', 'Pz', 'Cz', 'Fz', 'F7', 'F8']
fig, axn = plt.subplots(3, 5, sharex=True, sharey=True)
cbar_ax = fig.add_axes([.91, .3, .03, .4])
cmap = sns.diverging_palette(220, 10, as_cmap=True)
for i, ax in enumerate(axn.flat):
    mat_data = scipy.io.loadmat(eeg_epochs_paths[i], squeeze_me=True,
                                struct_as_record=False)['EEG']
    eeg_data, _ = truncate_epochs(mat_data, 100, remove_baseline_activity=True,
                                  modwt_ready=True, nlevels=7)
    subject_name = eeg_epochs_paths[i].split("/")[8]
    text_type = eeg_epochs_paths[i].split("/")[10]
    eeg_text_list = [epoch.textname for epoch in mat_data.epoch]
    chan_labels = [chanloc.labels for chanloc in mat_data.chanlocs]
    m = np.matrix([np.mean(trial, 0) for trial in eeg_data])
    m = m[:, [chan_labels.index(channel) for channel in channels]]
    sns.heatmap(m.T, ax=ax, cbar=i == 0, cbar_ax=None if i else cbar_ax,
                yticklabels=channels, cmap=cmap, robust=True)
    ax.set_title('%s - %s' % (subject_name, 'F'))
fig.tight_layout(rect=[0, 0, .9, 1])
plt.savefig('mean_per_chan_per_subj_per_trial.png')
plt.show()



# %% heatmap: Mean per chan per subj per trial
fig, axn = plt.subplots(3, 5, sharex=True, sharey=True)
cbar_ax = fig.add_axes([.91, .3, .03, .4])
cmap = sns.diverging_palette(220, 10, as_cmap=True)
for i, ax in enumerate(axn.flat):
    mat_data = scipy.io.loadmat(eeg_epochs_paths[i], squeeze_me=True,
                                struct_as_record=False)['EEG']
    eeg_data, _ = truncate_epochs(mat_data, 100, remove_baseline_activity=True,
                                  modwt_ready=True, nlevels=7)
    subject_name = eeg_epochs_paths[i].split("/")[8]
    text_type = eeg_epochs_paths[i].split("/")[10]
    eeg_text_list = [epoch.textname for epoch in mat_data.epoch]
    chan_labels = [chanloc.labels for chanloc in mat_data.chanlocs]
    m = np.matrix([np.mean(trial, 0) for trial in eeg_data])
    m = m[:, [chan_labels.index(channel) for channel in chan_labels]]
    sns.heatmap(m.T, ax=ax, cbar=i == 0, cbar_ax=None if i else cbar_ax,
                cmap=cmap, robust=True)
    ax.set_title('%s - %s' % (subject_name, 'F'))
fig.tight_layout(rect=[0, 0, .9, 1])
plt.savefig('mean_per_chan_per_subj_per_trial_2.png')
plt.show()





#%% heatmap: Variances per chan per subj per trial
channels = ['Oz', 'Pz', 'Cz', 'Fz', 'F7', 'F8']
fig, axn = plt.subplots(3, 5, sharex=True, sharey=True)
cbar_ax = fig.add_axes([.91, .3, .03, .4])
cmap = sns.diverging_palette(220, 10, as_cmap=True)
for i, ax in enumerate(axn.flat):
    mat_data = scipy.io.loadmat(eeg_epochs_paths[i], squeeze_me=True,
                                struct_as_record=False)['EEG']
    trunc_data = truncate_epochs(mat_data, 100, remove_baseline_activity=True,
                                 modwt_ready=False)
    subject_name = eeg_epochs_paths[i].split("/")[8]
    text_type = eeg_epochs_paths[i].split("/")[10]
    eeg_text_list = [epoch.textname for epoch in mat_data.epoch]
    chan_labels = [chanloc.labels for chanloc in mat_data.chanlocs]
    v = np.matrix([np.var(trial, 0) for trial in eeg_data])
    v = v[:, [chan_labels.index(channel) for channel in channels]]
    sns.heatmap(v.T, ax=ax, cbar=i == 0, cbar_ax=None if i else cbar_ax,
                yticklabels=channels, cmap=cmap, robust=True)
    ax.set_title(
        '%s - %s' %
        (subject_name, 'F'))
fig.tight_layout(rect=[0, 0, .9, 1])
plt.savefig('var_per_chan_per_subj_per_trial.png')
plt.show()


#%%
rdf = pd.read_excel('/home/bolivier/cw/ema/ema/share/data/HMM_feat_outfile_v6_165_166_viterbi_states.xls')

subject_names = rdf['SUBJ'].unique()
text_names = rdf['TEXT'].unique()
channel_names = ['Oz', 'Pz', 'Cz', 'Fz', 'F7', 'F8']
scale_names = ['Scale 1', 'Scale 2', 'Gamma h', 'Gamma l', 'Beta', 'Alpha', 'Theta']
phase_names = range(4)

nrows = len(subject_names) * len(text_names) * len(channel_names) * len(scale_names) * len(phase_names) * 2000  # avg len per trial per phase

def df_empty(index, columns, dtypes):
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        if d == np.uint16:
            df[c] = pd.Series([0] * len(index), dtype=np.uint16)
        else:
            df[c] = pd.Series(dtype=d)
    return df

df = df_empty(index=range(nrows),
              columns=['SUBJECT', 'TEXT', 'CHANNEL', 'SCALE', 'PHASE', 'TIME', 'VALUE'],
              dtypes=['category', 'category', 'category', 'category', 'category', np.uint16, np.float16])
df['SUBJECT'] = df['SUBJECT'].cat.set_categories(subject_names)
df['TEXT'] = df['TEXT'].cat.set_categories(text_names)
df['CHANNEL'] = df['CHANNEL'].cat.set_categories(channel_names)
df['SCALE'] = df['SCALE'].cat.set_categories(scale_names)
df['PHASE'] = df['PHASE'].cat.set_categories(phase_names)


df.loc[0] = pd.Series({'SUBJECT': 1, 'TEXT': 'chasse_oiseaux-f1','CHANNEL': 'Fz',
                       'SCALE': 'Scale 1', 'PHASE': 0, 'TIME': np.uint16(1), 'VALUE': 763.32})

print(df.memory_usage())





#%%
channels = ['Oz', 'Pz', 'Cz', 'Fz', 'F7', 'F8']
scales = ['Scale 1', 'Scale 2', 'Gamma h', 'Gamma l', 'Beta', 'Alpha', 'Theta']
nlevels = len(scales)
rdf = pd.read_excel('/home/bolivier/cw/ema/ema/share/data/HMM_feat_outfile_v6_165_166_viterbi_states.xls')
wt_per_phase_per_channel = {k: [] for k in range(4)}
eeg_epochs_paths = glob2.glob('/home/bolivier/cw/ema/eega/data/eeg-new/**/F/Trig_S1001_XLS/*.mat')
missing_eeg_records = []
for eeg_epochs_path in eeg_epochs_paths:
    mat_data = scipy.io.loadmat(eeg_epochs_path, squeeze_me=True,
                                struct_as_record=False)['EEG']
    eeg_data, censors = truncate_epochs(
        mat_data, 100, remove_baseline_activity=True, modwt_ready=True, nlevels=nlevels)
    # subject_name = eeg_epochs_paths[0].split("/")[8]
    subject_name = int(eeg_epochs_path.split("/")[8][1:])
    text_type = eeg_epochs_path.split("/")[9]
    eeg_text_list = [epoch.textname for epoch in mat_data.epoch]
    chan_labels = [chanloc.labels for chanloc in mat_data.chanlocs]
    print("opening file subj %s, text_type %s, path %s" %
          (subject_name, text_type, eeg_epochs_path))
    for em_text in rdf.loc[(rdf["SUBJ"] == subject_name),  # &
                                   # (rdf["TEXT_TYPE"] == text_type.lower()),
                                   "TEXT"].unique():
        if em_text not in eeg_text_list:
            missing_eeg_records.append(
                (subject_name, text_type, em_text))
        else:
            eeg_trial_number = eeg_text_list.index(em_text)
            trial = rdf.loc[(rdf["SUBJ"] == subject_name) &
                            # (rdf["TEXT_TYPE"] == text_type.lower()) &
                            (rdf["TEXT"] == em_text)]
            eeg_end_time = get_epoch_end_time(mat_data.epoch[eeg_trial_number], mat_data.times[-1])
            phases = trial_to_phases(trial, eeg_end_time)
            for phase in phases:
                wt_per_chan = []
                for channel in channels:
                    chan_id = chan_labels.index(channel)
                    d = eeg_data[eeg_trial_number][:, chan_id]
                    wt = rmodwt(robjects.FloatVector(d), wf="la8", n_levels=nlevels)
                    wt = rphaseshift(wt, 'la8')
                    wt = np.array(wt)
                    wt = wt[:-1, :]
                    if censors[eeg_trial_number, 1] == 0:
                        wt = wt[:, censors[eeg_trial_number, 0]:]
                    else:
                        wt = wt[:, censors[eeg_trial_number, 0]:-censors[eeg_trial_number, 1]]
                    wt = wt[:, phase[0]:phase[1]]
                    wt_per_chan.append(wt)
                wt_per_phase_per_channel[phase[2]].append(
                    np.dstack(wt_per_chan))

for key in wt_per_phase_per_channel.keys():
    wt_per_phase_per_channel[key] = np.concatenate(
        wt_per_phase_per_channel[key], axis=1)



#%% wt
plot_wt_var_per_chan_per_phase(wt_per_phase_per_channel)
plot_wt_cor_per_chan_per_phase(wt_per_phase_per_channel, 0)
plot_wt_cor_per_chan_per_phase(wt_per_phase_per_channel, 1)
plot_wt_cor_per_chan_per_phase(wt_per_phase_per_channel, 2)
plot_wt_cor_per_chan_per_phase(wt_per_phase_per_channel, 3)
plot_wt_cor_per_chan_per_phase(wt_per_phase_per_channel, 4)
plot_wt_cor_per_chan_per_phase(wt_per_phase_per_channel, 5)
plot_wt_cor_per_chan_per_phase(wt_per_phase_per_channel, 6)
# plot_wt_cor_per_chan_per_phase(wt_per_phase_per_channel, 7)


