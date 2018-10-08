library(openxlsx)  # read.xlsx
library(rmatio)  # read.mat
library(waveslim)  # modwt, phase.shift


eeg_data_path = "~/EEG-sample/s01-F.mat"
em_data_path = "~/cw/ema/ema/share/data/em-ry35.xlsx"


`%--%` = function(x, y) do.call(sprintf, c(list(x), y))

get_trial_duration = function(df, subject_name, text_name) {
  trial = df[which(df$TEXT == text_name & df$SUBJ_NAME == subject_name), ]
  n_fixations = dim(trial)[1]
  trial_duration = trial[n_fixations, "FIX_LATENCY"] - trial[1, "FIX_LATENCY"]
  return(trial_duration)
}

extract_relevent_eeg_data = function(eeg_data, trial_number, trial_duration, lag_duration) {
  return(eeg_data[, 1:trial_duration + lag_duration, trial_number])
}

eeg_s01_F = read.mat("~/EEG-sample/s01-F.mat")
em_s01_F = read.xlsx(em_data_path)
em_s01_F = em_s01_F[which(em_s01_F$SUBJ_NAME == "s01" & em_s01_F$TEXT_TYPE == "f"), ]



eeg_s01_F$ALLEEG$data

avg_eeg_s01_F = apply(eeg_s01_F$ALLEEG$data[[1]], c(1,2), mean)

ws = modwt(avg_eeg_s01_F[1,], wf="la8", n.levels=7, boundary='periodic')

ws = apply(avg_eeg_s01_F, )
