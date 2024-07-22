import matplotlib.pyplot as plt
import numpy as np
from MCMC import MCMCModel
from wave_funcs import damped_wave
from utility_funcs import generate_noise, evaluate_wave_fcn
from plotting_funcs import PlottingWrapper


if __name__ == "__main__":
    REAL_AMPLITUDE = 7.0
    REAL_DAMPING = 0.3
    REAL_ANGULAR_FREQ = 1.2
    REAL_THETA = np.array(
        [REAL_AMPLITUDE, REAL_DAMPING, REAL_ANGULAR_FREQ]
    )
    RANGES = np.array(
        [
            [0.1, 10.0],    # amplitude (A) range
            [0.1, 3.0],     # damping (b) range
            [0.1, 10.0],    # angular frequency (omega) range
        ]
    )
    
    WAVE_FCN = damped_wave
    DATA_STEPS = 1000
    TIMESPAN = 20
    WAVE_KWARGS = {"phase": 0.0, "seconds": TIMESPAN, "steps": DATA_STEPS}
    SNR = 1.0
    NUM_ITERATIONS = 10000
    THIN_PERCENTAGE = 0.7   # between 0 and 1. (0.1 would use 10% of the samples, 1.0 would use all of them)
    CUTOFF_START = 0.3      # percentage of the samples from the beginning that will be skipped
    
    #----------------------------------------------------------------------------
    real_wave = evaluate_wave_fcn(WAVE_FCN, REAL_THETA, WAVE_KWARGS)

    model = MCMCModel(function=WAVE_FCN, param_ranges=RANGES, function_kwargs=WAVE_KWARGS)
    noise = generate_noise(real_wave, snr=SNR)
    samples = model.metropolis_hastings(noise, NUM_ITERATIONS)

    #model.print_sample_statuses(samples)
    #----------------------------------- Plotting -------------------------------
    plotter = PlottingWrapper(
        samples=samples,
        param_ranges=RANGES,
        wave_fcn=WAVE_FCN,
        wave_kwargs=WAVE_KWARGS,
        real_parameters=REAL_THETA,
        noise=noise,
        snr=SNR
    )
    figure, axes = plotter.plot_mcmc_wave_results()
    plt.show()