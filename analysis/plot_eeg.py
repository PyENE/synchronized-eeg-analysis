# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import colorsys
import eega
import ema
import glob2
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import pywt
import seaborn as sns


def load_channel_names(
        channel_names_path=os.path.join(eega.DATA_PATH, 'channel_names.txt')):
    id2chan = {}
    with open(channel_names_path, 'r') as f:
        for line in f:
            elem = line.rstrip().split(",")
            id2chan[int(elem[0])] = elem[1]
    chan2id = {v: k for k, v in id2chan.items()}
    return id2chan, chan2id


def trial_to_phases(trial, max_trial_duration):
    phase_sequence = []
    for i in trial.index:
        if trial.loc[i, 'ISLAST']:
            t = trial.loc[i, 'FDUR']
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
    if len(phase_sequence) > max_trial_duration:
        max_trial_duration = phase_sequence[0:max_trial_duration]
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


def load_rgb_colors(n_colors=4):
    hues = np.linspace(0, 1, n_colors + 1)[:-1]
    color_list = [(h_i, 0.60, 0.65) for h_i in hues]
    rgb_colors = []
    for color in color_list:
        rgb_colors.append(colorsys.hsv_to_rgb(color[0], color[1], color[2]))
    return rgb_colors


def plot_eeg2(time_series):
    levels = 6
    tmax = len(time_series) / 2**levels
    tmax *= 2**levels
    ws = pywt.swt(time_series[0:tmax], 'db8', levels)
    ws = np.stack(ws)[:, 1, :]
    """
    ws = eega.modwt(time_series[0:tmax], 'db8', 6)
    """
    f, axarr = plt.subplots(ws.shape[0] + 1, sharex=True)
    ylabels = ["s" + str(scale + 1)
               for scale in range(0, ws.shape[0])]
    plt.xlabel('time (ms)')
    for i in range(0, ws.shape[0]):
        axarr[i].plot(range(0, tmax),
                      ws[ws.shape[0] - i - 1, :],
                      linewidth=1)
        axarr[i].set_ylabel("%s" % ylabels[ws.shape[0] - i - 1], rotation=0)
        # axarr[i].set_yticklabels([])
    axarr[ws.shape[0]].plot(range(0, tmax), time_series[0:tmax], linewidth=1)
    axarr[ws.shape[0]].set_ylabel('TS', rotation=0)
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.show()


def plot_time_series(time_series, phases, ylabels):
    """@TODO: inherit from plt for settings labels"""
    rgb_colors = load_rgb_colors(n_colors=4)
    f, axarr = plt.subplots(time_series.shape[0], sharex=True)
    plt.xlabel("time (ms)")
    for i in range(0, time_series.shape[0]):
        for phase in phases:
            axarr[i].plot(range(phase[0], phase[1]),
                          time_series[i, phase[0]:phase[1]],
                          color=rgb_colors[phase[2]],
                          linewidth=1)
        axarr[i].set_ylabel("%s" % ylabels[i], rotation=0)
        # axarr[i].set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.show()


def plot_eeg(eeg_data, channels, phases):
    _, chan2id = load_channel_names()
    channels_id = [chan2id[channel] for channel in channels]
    channel_data = eeg_data[channels_id, :]
    ylabels = [channel + " (uv)" for channel in channels]
    plot_time_series(channel_data, phases, ylabels)


def plot_wavelet_coefficients(ws, phases):
    ylabels = ["s" + str(scale + 1) for scale in range(0, ws.shape[0])]
    plot_time_series(ws, phases, ylabels)


def plot_mra(ws, phases):
    ylabels = ["s" + str(scale + 1) for scale in range(0, ws.shape[0])]
    plot_time_series(ws, phases, ylabels)


def plot_ws_var_per_chan_per_phase(ws_per_phase_per_channel):
    """per text type, per subject within eeg_data"""
    channels = ['Oz', 'Pz', 'Cz', 'Fz', 'F7', 'F8']
    scales = ['Scale 1', 'Scale 2', 'Gamma h',
              'Gamma l', 'Beta', 'Alpha', 'Theta']
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, axn = plt.subplots(2, 2, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for i, ax in enumerate(axn.flat):
        v = np.var(ws_per_phase_per_channel[i], 1)
        sns.heatmap(np.flip(v, 0), ax=ax,
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax,
                    yticklabels=scales, xticklabels=channels, cmap=cmap)
        ax.set_title('Strategy ' + str(i))
    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.show()


def plot_ws_cor_per_chan_per_phase(ws_per_phase_per_channel, scale):
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
        c = np.corrcoef(ws_per_phase_per_channel[i][scale, :, :].T)
        mask = np.zeros_like(c, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(c, ax=ax, mask=mask, center=0,  vmax=0.8, vmin=0,
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax,
                    yticklabels=ychannels, xticklabels=xchannels, cmap=cmap)
        ax.set_title('Strategy ' + str(i))
    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.show()


def plot_correlation_matrix(corr, channels):
    xchannels = channels[:]
    xchannels[-1] = ''
    ychannels = channels[:]
    ychannels[0] = ''
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    with sns.axes_style("white"):
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0,
                    square=True, linewidths=.5, cbar=True,
                    cbar_kws={"shrink": .5}, xticklabels=xchannels,
                    yticklabels=ychannels)
    plt.show()


def get_trial_time(trial):
    return (trial.loc[trial['ISLAST'] == 1, 'REL_FIX_TIME'].values[0] -
            trial.loc[trial['ISFIRST'] == 1, 'REL_FIX_TIME'].values[0])


def normalize_ws(ws, method):
    if method == 1:
        return np.divide(np.square(ws).T,
                         (np.arange(levels, 0, -1))).T
    elif method == 2:
        return np.divide(ws.T, np.sqrt(np.arange(levels, 0, -1))).T
    elif method == 3:
        return np.divide(ws.T, np.square(np.arange(levels, 0, -1))).T
    elif method == 4:
        return np.divide(ws.T, np.arange(levels, 0, -1)).T


# def main():
plt.style.use('seaborn')
id2chan, chan2id = load_channel_names()
channels = ['Oz', 'Pz', 'Cz', 'Fz', 'F7', 'F8']
scales = ['Scale 1', 'Scale 2', 'Gamma h', 'Gamma l', 'Beta', 'Alpha', 'Theta']
levels = len(scales)

"""
subject_name = "s01"
text_type = "F"
trial_number = 15

eeg_data = mne.io.read_epochs_eeglab(
    os.path.join(eega.DATA_PATH, "EEG", subject_name, "SynchroTag14",
                    text_type, "Trig_S1001",
                    "synchro_s01_test.set"))  # .get_data() * 1e6  # uV
"""


"""
# plot covariance matrix
eeg_data.info['ch_names'] = id2chan.values()
eeg_data._data = eeg_data._data * 1e6
for ch in eeg_data.info['chs']:
    ch['ch_name'] = id2chan[int(ch['ch_name'])]
noise_cov = mne.compute_covariance(eeg_data, tmax=0)
noise_cov.plot(eeg_data.info, proj=True)
"""


"""
# plot wavelet coefficients
rdf = pd.read_excel(os.path.join(ema.DATA_PATH, 'em-ry35.xlsx'))
trial = rdf.loc[(rdf["SUBJ_NAME"] == subject_name) &
                (rdf["TEXT_TYPE"] == text_type.lower()) &
                (rdf["TEXT_NO"] == trial_number)]

phases = trial_to_phases(trial, eeg_data.shape[2])
plot_eeg(eeg_data[trial_number, :, :], channels, phases)
ws = eega.modwt(
    eeg_data[trial_number, chan2id[channels[0]], 0:phases[-1][1]], 'db8', 6)
plot_wavelet_coefficients(ws, phases)
ws = eega.modwt(eeg_data[trial_number, chan2id[channels[0]], :], 'db8', 6)
plot_wavelet_coefficients(ws, phases)

# mra = eega.modwtmra(ws, 'db2')
# plot_mra(mra, phases)
"""


"""
# concatenate epochs, average EEGs per chan, compute var/cov
eeg_data = np.zeros((32, 15647))
eeg_epochs_paths = glob2.glob(
    os.path.join(eega.DATA_PATH, 'EEG', '**/Trig_S1001/**/*.set'))
i = 0
for eeg_epochs_path in eeg_epochs_paths:
    i += 1
    if i > 5:
        break
    if (eeg_data == 0).all():
        eeg_data = mne.io.read_epochs_eeglab(
            eeg_epochs_path).average().data * 1e6
    else:
        eeg_data += mne.io.read_epochs_eeglab(
            eeg_epochs_path).average().data * 1e6
eeg_data /= i
eeg_ws_channels = []
for channel in channels:
    # eeg_ws_channels.append(eega.modwt(
    #    eeg_data[chan2id[channel], :], 'db8', 6))
    tmax = len(eeg_data[chan2id[channel], :]) / 2**levels
    tmax *= 2**levels
    ws = (eeg_data[chan2id[channel], :][0:tmax], 'db8', levels)
    ws = np.stack(ws)[:, 1, :]
    ws = ws.T
    ws = np.flip(ws, 1)
    eeg_ws_channels.append(ws)
eeg_ws_channels = np.stack(eeg_ws_channels)
v = np.var(eeg_ws_channels, 1)
sns.heatmap(v.T, yticklabels=scales, xticklabels=channels)
plt.show()
c = np.corrcoef(eeg_ws_channels[:, :, -1])
plot_correlation_matrix(c, channels)
"""


"""
# map (subject_name, text_type, trial_id) to id in eeg_epochs.get_data()
trial2id = {}
nb_trials = 0
for eeg_epochs_path in eeg_epochs_paths:
    subject_name = eeg_epochs_paths[0].split("/")[8]
    text_type = eeg_epochs_paths[0].split("/")[10]
    eeg_epoch = mne.io.read_epochs_eeglab(eeg_epochs_path)
    for i in range(0, eeg_epoch.get_data().shape[0]):
        trial2id[(subject_name, text_type, i)
                    ] = nb_trials + i
    eeg_epochs.append(eeg_epoch)
    nb_trials += eeg_epoch.get_data().shape[0]
    # to be deleted
    if nb_trials > 200:
        break
eeg_epochs = mne.concatenate_epochs(eeg_epochs)
noise_cov = mne.compute_covariance(eeg_epochs)
noise_cov.plot(eeg_epochs.info, proj=True)
eeg_epochs.average().plot(time_unit='ms')
"""

levels = 7
# tmax = len(time_series) / 2**levels
# tmax *= 2**levels
# tmax = 15616
# rdf = pd.read_excel(os.path.join(ema.DATA_PATH, 'em-ry35.xlsx'))
rdf = pd.read_excel(os.path.join(
    ema.DATA_PATH, 'HMM_feat_outfile_v6_165_166_viterbi_states.xls'))
ws_per_phase_per_channel = {k: [] for k in range(4)}
eeg_data = np.zeros((32, 15647))
eeg_epochs_paths = glob2.glob(
    os.path.join(eega.DATA_PATH, 'EEG', '**/F/Trig_S1001/**/*.set'))
missing_eeg_records = []
for eeg_epochs_path in eeg_epochs_paths:
    epoch = mne.io.read_epochs_eeglab(eeg_epochs_path)
    eeg_data = epoch.get_data()
    # subject_name = eeg_epochs_paths[0].split("/")[8]
    subject_name = int(eeg_epochs_path.split("/")[8][1:])
    text_type = eeg_epochs_path.split("/")[10]
    print("opening file %s\nsubj %s, text_type %s" %
          (eeg_epochs_path, subject_name, text_type))
    for em_trial_number in rdf.loc[(rdf["SUBJ"] == subject_name),  # &
                                   # (rdf["TEXT_TYPE"] == text_type.lower()),
                                   "TEXT_NO"].unique():
        trial = rdf.loc[(rdf["SUBJ"] == subject_name) &
                        # (rdf["TEXT_TYPE"] == text_type.lower()) &
                        (rdf["TEXT_NO"] == em_trial_number)]
        trial_time = get_trial_time(trial)
        tmax = trial_time / 2**levels
        tmax *= 2**levels
        if em_trial_number not in epoch.event_id.values():
            missing_eeg_records.append(
                (subject_name, text_type, em_trial_number))
        else:
            eeg_trial_number = epoch.event_id.values().index(em_trial_number)
            phases = trial_to_phases(trial, tmax)
            for phase in phases:
                ws_per_chan = []
                for channel in channels:
                    d = eeg_data[eeg_trial_number - 1,
                                 chan2id[channel] - 1, 0:tmax]
                    if np.var(d) > 10000:
                        print('subj %s, text %s, trial %s, channel %s, var %s' % (
                            subject_name, text_type, eeg_trial_number, channel,
                            np.var(d)))
                    ws = pywt.swt(d, 'sym8', levels)
                    ws = np.stack(ws)[:, 1, :]
                    ws = ws[:, phase[0]:phase[1]]
                    ws_per_chan.append(ws)
                ws_per_phase_per_channel[phase[2]].append(
                    np.dstack(ws_per_chan))

for key in ws_per_phase_per_channel.keys():
    ws_per_phase_per_channel[key] = np.concatenate(
        ws_per_phase_per_channel[key], axis=1)


plot_ws_var_per_chan_per_phase(ws_per_phase_per_channel)

plot_ws_cor_per_chan_per_phase(ws_per_phase_per_channel, 0)
plot_ws_cor_per_chan_per_phase(ws_per_phase_per_channel, 1)
plot_ws_cor_per_chan_per_phase(ws_per_phase_per_channel, 2)
plot_ws_cor_per_chan_per_phase(ws_per_phase_per_channel, 3)
plot_ws_cor_per_chan_per_phase(ws_per_phase_per_channel, 4)
plot_ws_cor_per_chan_per_phase(ws_per_phase_per_channel, 5)
plot_ws_cor_per_chan_per_phase(ws_per_phase_per_channel, 6)

"""
if __name__ == '__main__':
    main()
"""


# %%
eeg_epochs_paths = glob2.glob(
    os.path.join(eega.DATA_PATH, 'EEG', '**/F/Trig_S1001/**/*.set'))
fig, axn = plt.subplots(3, 5, sharex=True, sharey=True)
cbar_ax = fig.add_axes([.91, .3, .03, .4])
cmap = sns.diverging_palette(220, 10, as_cmap=True)
for i, ax in enumerate(axn.flat):
    subject_name = int(eeg_epochs_paths[i].split("/")[8][1:])
    text_type = eeg_epochs_paths[i].split("/")[10]
    epoch = mne.io.read_epochs_eeglab(eeg_epochs_paths[i])
    eeg_data = epoch.get_data(
    )[:, [chan2id[channel] - 1 for channel in channels], :]
    v = np.var(eeg_data, 2)
    sns.heatmap(v, ax=ax, cbar=i == 0, cbar_ax=None if i else cbar_ax,
                xticklabels=channels, cmap=cmap, robust=True)
    ax.set_title(
        '%s - %s' %
        (subject_name, text_type))
fig.tight_layout(rect=[0, 0, .9, 1])
plt.show()

# %%

epoch.plot_image(picks=[chan2id[channel] - 1 for channel in channels])
epoch.plot(picks=[1], n_epochs=10)


bv = mne.io.read_raw_brainvision(
    '/home/bolivier/EEG-sample/synchro_s01_test_BV.vhdr')
