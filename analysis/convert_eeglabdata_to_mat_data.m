for t = ['A' 'F' 'M']
    for s = {'s01' 's02' 's04' 's05' 's06' 's07' 's08' 's10' 's13' 's14' 's17' 's18' 's19' 's20' 's21'}
       EEG = pop_loadset(strcat('/home/bolivier/cw/ema/eega/data/eeg-new/', s{1}, '/', t, '/Trig_S1001_XLS/synchro_', s{1},'_test.set'))
       save(strcat('/home/bolivier/cw/ema/eega/data/eeg-new/', s{1}, '/', t, '/Trig_S1001_XLS/synchro_', s{1},'_test.mat'), 'EEG')
    end
end
