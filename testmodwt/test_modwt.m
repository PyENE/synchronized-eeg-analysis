gdpdata = csvread('../GDPcomponents.csv', 1)

n_levels = 5
modwt_len = size(gdpdata, 1)- mod(size(gdpdata, 1), 2^n_levels)
ts = gdpdata(1:modwt_len, 3)

wt = modwt(ts, n_levels, 'sym8')

figure;
subplot(7,1,6);
plot(wt(1,:));
subplot(7,1,5);
plot(wt(2,:));
subplot(7,1,4);
plot(wt(3,:));
subplot(7,1,3);
plot(wt(4,:));
subplot(7,1,2);
plot(wt(5,:));
subplot(7,1,1);
plot(wt(6,:));
subplot(7,1,7);
plot(ts);
saveas(gcf,'~/cw/ema/eega/test/matlab_modwt_wt_periodic.jpg')


wt = modwt(ts, n_levels, 'sym8', 'reflective')
%wt = wt(:, 1:modwt_len)

figure;
subplot(7,1,6);
plot(wt(1,:));
subplot(7,1,5);
plot(wt(2,:));
subplot(7,1,4);
plot(wt(3,:));
subplot(7,1,3);
plot(wt(4,:));
subplot(7,1,2);
plot(wt(5,:));
subplot(7,1,1);
plot(wt(6,:));
subplot(7,1,7);
plot(ts);

saveas(gcf,'~/cw/ema/eega/test/matlab_modwt_wt_reflective.jpg')