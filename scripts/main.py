import matplotlib.pyplot as plt
import numpy as np
from MCMC import MCMCModel
from wave_func import mcmc_wave

REAL_AMPLITUDE = 7.0
REAL_DAMPING = 0.3
REAL_ANGULAR_FREQ = 1.2
NOISE_AMPLITUDE = 0.0
NOISE_SCALE = 0.0
NUM_ITERATIONS = 100000
DATA_STEPS = 1000
TIMESPAN = 20
THIN_PERCENTAGE = 1.0 # between 0 and 1. (0.1 would use 10% of the samples, 1.0 would use all of them)
CUTOFF_START = 0.2 # percentage of the samples from the beginning that will be skipped

wave_kwargs = {"phase": 0.0, "seconds": TIMESPAN, "steps": DATA_STEPS}
time, real_wave = mcmc_wave(REAL_AMPLITUDE, REAL_DAMPING, REAL_ANGULAR_FREQ, return_time=True, **wave_kwargs)

ranges = np.array(
    [
        [0.1, 10.0],    # amplitude (A) range
        [0.1, 3.0],     # damping (b) range
        [0.1, 10.0],    # angular frequency (omega) range
    ]
)

model = MCMCModel(function=mcmc_wave, param_ranges=ranges, function_kwargs=wave_kwargs)
samples = model.metropolis_hastings(real_wave, NUM_ITERATIONS, noise_scale=NOISE_SCALE, noise_amplitude=NOISE_AMPLITUDE)
noise = model.generate_noise(real_wave, scale=NOISE_SCALE, noise_amp=NOISE_AMPLITUDE)

#model.print_sample_statuses(samples)

amplitude_samples = samples[:, 0]
damping_samples = samples[:, 1]
angular_freq_samples = samples[:, 2]

amplitude_samples = model.thin_samples(amplitude_samples, thin_percentage=THIN_PERCENTAGE, cutoff_percentage=CUTOFF_START)
damping_samples = model.thin_samples(damping_samples, thin_percentage=THIN_PERCENTAGE, cutoff_percentage=CUTOFF_START)
angular_freq_samples = model.thin_samples(angular_freq_samples, thin_percentage=THIN_PERCENTAGE, cutoff_percentage=CUTOFF_START)

mean_amplitude = np.mean(amplitude_samples)
mean_damping = np.mean(damping_samples)
mean_angular_freq = np.mean(angular_freq_samples)

std_amplitude = np.std(amplitude_samples)
std_damping = np.std(damping_samples)
std_angular_freq = np.std(angular_freq_samples)

result_wave = mcmc_wave(mean_amplitude, mean_damping, mean_angular_freq, **wave_kwargs)
figure = plt.figure(figsize=(13, 7), layout="constrained")
figure.suptitle("Extracting Signal Parameters From Noise")
axes = figure.subplot_mosaic(
"""
WWN.
ABF.
""",
width_ratios=[1, 1, 1, 0.05]
)

axes["A"].hist(amplitude_samples, bins=100, density=True, label="estimated distribution")
axes["A"].set_xlabel("Amplitude")
ylim = axes["A"].get_ylim()
axes["A"].plot([REAL_AMPLITUDE, REAL_AMPLITUDE], [0.0, ylim[1]], label="true value")
axes["A"].plot([mean_amplitude, mean_amplitude], [0.0, ylim[1]], label="sample mean")
axes["A"].plot([mean_amplitude+std_amplitude, mean_amplitude+std_amplitude], [0.0, ylim[1]], "r--", label="+1 std")
axes["A"].plot([mean_amplitude-std_amplitude, mean_amplitude-std_amplitude], [0.0, ylim[1]], "r--", label="-1 std")
axes["A"].legend(prop={"size": 8})
axes["A"].set_xlim(left=ranges[0, 0], right=ranges[0, 1])


axes["B"].hist(damping_samples, bins=100, density=True, label="estimated distribution")
axes["B"].set_xlabel("Damping")
ylim = axes["B"].get_ylim()
axes["B"].plot([REAL_DAMPING, REAL_DAMPING], [0.0, ylim[1]], label = "true value")
axes["B"].plot([mean_damping, mean_damping], [0.0, ylim[1]], label="sample mean")
axes["B"].plot([mean_damping+std_damping, mean_damping+std_damping], [0.0, ylim[1]], "r--", label="+1 std")
axes["B"].plot([mean_damping-std_damping, mean_damping-std_damping], [0.0, ylim[1]], "r--", label="-1 std")
axes["B"].legend(prop={"size": 8})
axes["B"].set_xlim(left=ranges[1, 0], right=ranges[1, 1])

axes["F"].hist(angular_freq_samples, bins=100, density=True, label="estimated distribution")
axes["F"].set_xlabel("Angular frequency")
ylim = axes["F"].get_ylim()
axes["F"].plot([REAL_ANGULAR_FREQ, REAL_ANGULAR_FREQ], [0.0, ylim[1]], label = "true value")
axes["F"].plot([mean_angular_freq, mean_angular_freq], [0.0, ylim[1]], label="sample mean")
axes["F"].plot([mean_angular_freq+std_angular_freq, mean_angular_freq+std_angular_freq], [0.0, ylim[1]], "r--", label="+1 std")
axes["F"].plot([mean_angular_freq-std_angular_freq, mean_angular_freq-std_angular_freq], [0.0, ylim[1]], "r--", label="-1 std")
axes["F"].legend(prop={"size": 8})
axes["F"].set_xlim(left=ranges[2, 0], right=ranges[2, 1])
    
axes["W"].plot(time, real_wave, "r", label="real")
axes["W"].set_xlabel("time")
rms_deviation = model.compute_rms(real_wave, result_wave)
axes["W"].annotate(f"Real Parameters:\nAmplitude: {REAL_AMPLITUDE}\nDamping: {REAL_DAMPING}\nAngular Frequency: {REAL_ANGULAR_FREQ}\n\nRMS Deviation: {rms_deviation:.3f}", xy=(0.75, 0.6), xycoords="axes fraction")
axes["W"].plot(time, result_wave, "g", label="generated")
axes["W"].legend()

axes["N"].plot(time, noise)
axes["N"].set_xlabel("time")
axes["N"].annotate(f"Noise amplitude: {NOISE_AMPLITUDE}", xy=(0.53, 0.9), xycoords="axes fraction")

plt.show()