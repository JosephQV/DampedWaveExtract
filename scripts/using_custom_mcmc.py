import matplotlib.pyplot as plt
import numpy as np
from MCMC import MCMCModel
from wave_funcs import mcmc_wave
from utility_funcs import generate_noise
from plotting_funcs import plot_mcmc_wave_results

if __name__ == "__main__":
    REAL_AMPLITUDE = 7.0
    REAL_DAMPING = 0.3
    REAL_ANGULAR_FREQ = 1.2
    NOISE_AMPLITUDE = 1.0
    NOISE_SCALE = 1.0
    NUM_ITERATIONS = 10000
    DATA_STEPS = 1000
    TIMESPAN = 20
    THIN_PERCENTAGE = 0.7 # between 0 and 1. (0.1 would use 10% of the samples, 1.0 would use all of them)
    CUTOFF_START = 0.3 # percentage of the samples from the beginning that will be skipped

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
    noise = generate_noise(real_wave, scale=NOISE_SCALE, noise_amp=NOISE_AMPLITUDE)

    #model.print_sample_statuses(samples)

    figure, axes = plot_mcmc_wave_results(samples=samples, real_data=real_wave, real_params=np.array([REAL_AMPLITUDE, REAL_DAMPING, REAL_ANGULAR_FREQ]), param_ranges=ranges, wave_kwargs=wave_kwargs, noise=noise, x=time)
    plt.show()