library(waveslim)


`%--%` <- function(x, y) do.call(sprintf, c(list(x), y))


gdpdata = read.csv("~/cw/ema/eega/data/GDPcomponents.csv")
n_levels = 5
modwt_len = dim(gdpdata)[1] - dim(gdpdata)[1] %% 2^n_levels
ts = gdpdata[1:modwt_len, 'govtexp']

ws_wt_p = waveslim::modwt(ts, wf="la8", n.levels=n_levels, boundary='periodic')
ws_wt_p = phase.shift(ws_wt_p, 'la8')

jpeg('~/cw/ema/eega/test/r_waveslim_wt_periodic.jpg')
par(mfrow=c(7,1), mar=c(2,3,1,1))
plot(ws_wt_p$s5, type="l", ylab='')
mtext("mean=%.1e, var=%.1e" %--% c(mean(ws_wt_p$s5), var(ws_wt_p$s5)))
plot(ws_wt_p$d5, type="l")
mtext("mean=%.1e, var=%.1e" %--% c(mean(ws_wt_p$d5), var(ws_wt_p$d5)))
plot(ws_wt_p$d4, type="l")
mtext("mean=%.1e, var=%.1e" %--% c(mean(ws_wt_p$d4), var(ws_wt_p$d4)))
plot(ws_wt_p$d3, type="l")
mtext("mean=%.1e, var=%.1e" %--% c(mean(ws_wt_p$d3), var(ws_wt_p$d3)))
plot(ws_wt_p$d2, type="l")
mtext("mean=%.1e, var=%.1e" %--% c(mean(ws_wt_p$d2), var(ws_wt_p$d2)))
plot(ws_wt_p$d1, type="l")
mtext("mean=%.1e, var=%.1e" %--% c(mean(ws_wt_p$d1), var(ws_wt_p$d1)))
plot(ts, type="l")
mtext("mean=%.1e, var=%.1e" %--% c(mean(ts), var(ts)))
dev.off()

ws_wt_r = waveslim::modwt(ts, wf="la8", n.levels=n_levels, boundary='reflection')
ws_wt_r = phase.shift(ws_wt_r, 'la8')

jpeg('~/cw/ema/eega/test/r_waveslim_wt_refective.jpg')
par(mfrow=c(7,1), mar=c(2,3,1,1))
plot(ws_wt_r$s5[1:length(ts)], type="l", ylab='')
mtext("mean=%.1e, var=%.1e" %--% c(mean(ws_wt_r$s5), var(ws_wt_r$s5)))
plot(ws_wt_r$d5[1:length(ts)], type="l")
mtext("mean=%.1e, var=%.1e" %--% c(mean(ws_wt_r$d5), var(ws_wt_r$d5)))
plot(ws_wt_r$d4[1:length(ts)], type="l")
mtext("mean=%.1e, var=%.1e" %--% c(mean(ws_wt_r$d4), var(ws_wt_r$d4)))
plot(ws_wt_r$d3[1:length(ts)], type="l")
mtext("mean=%.1e, var=%.1e" %--% c(mean(ws_wt_r$d3), var(ws_wt_r$d3)))
plot(ws_wt_r$d2[1:length(ts)], type="l")
mtext("mean=%.1e, var=%.1e" %--% c(mean(ws_wt_r$d2), var(ws_wt_r$d2)))
plot(ws_wt_r$d1[1:length(ts)], type="l")
mtext("mean=%.1e, var=%.1e" %--% c(mean(ws_wt_r$d1), var(ws_wt_r$d1)))
plot(ts, type="l")
mtext("mean=%.1e, var=%.1e" %--% c(mean(ts), var(ts)))
dev.off()

