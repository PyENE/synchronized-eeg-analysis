# EM-synchronized EEG analysis using MODWT
<!---

## MODWT
Using R waveslim package interfaced with rpy2

## Project architecture

* analysis:
    * baseline_removal_effect.py
    * convert_eeglabdata_to_mat_data.m
    * lag_nature.py
    * plot_channel_correlations.py
    * plot_topomaps.py
    * plot_var_per_scale.py
* clusscript: scripts to be ran on cluster
* data: (stored somewhere else!)
    * corrmat: per channel per scale per phase per subject correlations
    * eeg: raw data as .mat files
        * s01/A/Trig_S1001_XLS/synchro_s01_test.mat
        * s01/M/Trig_S1001_XLS/synchro_s01_test.mat
        * s01/F/Trig_S1001_XLS/synchro_s01_test.mat
        * ...
        * s21/F/Trig_S1001_XLS/synchro_s01_test.mat
    * wt: wavelet coefficients
    * wt-gb: variance-aggregated wavelet coefficients
* eega:
    * modwt.py: wraps R waveslim MODWT and adds visualization methods
    * sync_eeg.py: Synchronize EEGs with EM data
* example

-->