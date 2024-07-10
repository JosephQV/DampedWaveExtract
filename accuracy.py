import matplotlib.pyplot as plt
import numpy as np
from MCMC import MCMCModel
from wave_func import mcmc_wave

def run_trial(model, real_amp, real_damp, real_angf, steps, timespan, thin_percentage, cutoff, num_iterations, noise_amplitude, noise_scale):
    # use given params to make a real wave
    real_wave = mcmc_wave(real_amp, real_damp, real_angf, seconds=timespan, steps=steps)
    # run metropolis hastings for the given number of iterations and given noise
    samples = model.metropolis_hastings(data=real_wave, num_iterations=num_iterations, noise_scale=noise_scale, noise_amplitude=noise_amplitude)
    # get the mean of the samples returned for each parameter, create the wave generated from them
    mean_amplitude = np.mean(model.thin_samples(samples[:,0], thin_percentage, cutoff))
    mean_damping = np.mean(model.thin_samples(samples[:,1], thin_percentage, cutoff))
    mean_angular_freq = np.mean(model.thin_samples(samples[:,2], thin_percentage, cutoff))
    generated_wave = mcmc_wave(mean_amplitude, mean_damping, mean_angular_freq, seconds=timespan, steps=steps)
    # find the rms between the real and generated waves
    rms = model.compute_rms(observed=real_wave, predicted=generated_wave)
    return rms

REAL_AMPLITUDE = 7.0
REAL_DAMPING = 0.3
REAL_ANGULAR_FREQ = 1.2
NOISE_AMPLITUDE = np.linspace(0.0, 7.5, 75)
NOISE_SCALE = 2.0
NUM_ITERATIONS = 10000
DATA_STEPS = 10000
TIMESPAN = 20
THIN_PERCENTAGE = 0.5 # between 0 and 1. (0.1 would use 10% of the samples, 1.0 would use all of them)
CUTOFF_START = 0.2 # percentage of the samples from the beginning that will be skipped

ranges = np.array(
    [
        [0.1, 10.0],    # amplitude (A) range
        [0.1, 3.0],     # damping (b) range
        [0.1, 10.0],    # angular frequency (omega) range
    ]
)
kwargs = {"seconds": TIMESPAN, "steps": DATA_STEPS, "return_time": False}

model = MCMCModel(mcmc_wave, param_ranges=ranges, function_kwargs=kwargs)

accuracy_by_noise = []
for noise in NOISE_AMPLITUDE:
    rms = run_trial(model, REAL_AMPLITUDE, REAL_DAMPING, REAL_ANGULAR_FREQ, DATA_STEPS, TIMESPAN, THIN_PERCENTAGE, CUTOFF_START, NUM_ITERATIONS, noise_amplitude=noise, noise_scale=NOISE_SCALE)
    accuracy_by_noise.append(rms)
figure = plt.figure()
axes = figure.add_subplot()
axes.scatter(NOISE_AMPLITUDE, accuracy_by_noise)
axes.set_xlabel("Noise Amplitude")
axes.set_ylabel("Accuracy (RMS)")
figure.suptitle("Wave Estimation Accuracy as a Function of Noise")

plt.show()