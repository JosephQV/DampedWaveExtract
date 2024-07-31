import numpy as np
import matplotlib.pyplot as plt
from wave_funcs import *
from utility_funcs import generate_noise, evaluate_wave_fcn
from plotting_funcs import FACECOLOR

wave_fcn = damped_wave
wave_theta = [20.0, 0.15, 1.5]
wave_kwargs = {"steps": 10000, "seconds": 30, "phase": 1.0 * np.pi / 2}
snrs = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
facecolor = FACECOLOR
save_path = "/home/joseph/LocalProjects/DampedWaveExtract/figures/poster"

figure = plt.figure(figsize=(16, 10), dpi=150)
axes = figure.add_axes(rect=(0,0,1,1), facecolor=facecolor)
axes.set_axis_off()

time, wave = evaluate_wave_fcn(damped_wave, wave_theta, wave_kwargs, return_time=True)
i = len(snrs)
for snr in snrs:
    axes.clear()
    noise = generate_noise(data=wave, snr=snr)
    axes.plot(time, noise, color="#1c0d7a", lw=0.7)
    axes.plot(time, wave, color="#911736", lw=2.5)
    figure.savefig(f"{save_path}/noise_level_{i}")
    i -= 1
    
