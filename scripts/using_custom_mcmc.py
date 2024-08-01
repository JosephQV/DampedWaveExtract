import numpy as np
from MCMC import MCMCModel
from wave_funcs import damped_wave
from utility_funcs import generate_noise, evaluate_wave_fcn, thin_samples
from plotting_funcs import PlottingWrapper
import sys


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
            [0.1, 10.0],     # damping (b) range
            [0.1, 10.0],    # angular frequency (omega) range
        ]
    )
    
    WAVE_FCN = damped_wave
    DATA_STEPS = 1000
    TIMESPAN = 25
    WAVE_KWARGS = {"phase": 0.0, "seconds": TIMESPAN, "steps": DATA_STEPS}
    SNR = 1.0
    NUM_ITERATIONS = 100000
    THIN_BY = 1
    BURN_PERCENTAGE = 0.0
    
    #----------------------------------------------------------------------------
    real_wave = evaluate_wave_fcn(WAVE_FCN, REAL_THETA, WAVE_KWARGS)

    model = MCMCModel(wave_fcn=WAVE_FCN, param_ranges=RANGES, wave_kwargs=WAVE_KWARGS)
    noise = generate_noise(real_wave, snr=SNR)
    samples, likelihoods = model.metropolis_hastings(noise, NUM_ITERATIONS)
    
    i = np.argmax(likelihoods)
    best_theta = samples[i]
    
    thin_samples(samples, THIN_BY, BURN_PERCENTAGE)

    #----------------------------------- Plotting -------------------------------
    outdir = sys.argv[1]
    
    plotter = PlottingWrapper(
        samples=samples,
        real_theta=REAL_THETA,
        best_theta=best_theta,
        ranges=RANGES,
        names=["Amplitude", "Damping", "Angular Frequency"],
        labels=["$A$", "$\\beta$", "$\\omega$"],
        wave_fcn=WAVE_FCN,
        wave_kwargs=WAVE_KWARGS,
        snr=SNR
    )
    figure, axes = plotter.plot_mcmc_wave_results()
    figure.savefig(f"{outdir}/CustomMCMCResults.png")
    
    figure, axes = plotter.plot_posterior_traces(single_chain=samples, iter_start=2000, iter_end = 2500)
    figure.savefig(f"{outdir}/SampleTraces.png")
