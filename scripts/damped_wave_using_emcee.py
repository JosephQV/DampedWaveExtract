import emcee
import numpy as np
import matplotlib.pyplot as plt
from wave_funcs import damped_wave
from utility_funcs import generate_noise, save_figure
from plotting_funcs import PlottingWrapper
from emcee_funcs import *
import sys
import time
import multiprocessing

    
if __name__ == "__main__":
    # -------------------------------------
    REAL_AMPLITUDE = 7.0
    REAL_DAMPING = 0.3
    REAL_ANGULAR_FREQ = 1.2
    REAL_THETA = np.array(
        [REAL_AMPLITUDE, REAL_DAMPING, REAL_ANGULAR_FREQ]
    )
    
    RANGES = np.array(
        [
            [0.1, 20.0],    # amplitude (A) range
            [0.1, 20.0],    # damping (b) range
            [0.1, 20.0],    # angular frequency (omega) range
        ]
    )
    
    WAVE_FCN = damped_wave
    DATA_STEPS = 1000
    TIMESPAN = 20
    WAVE_KWARGS = {"phase": 0.0, "seconds": TIMESPAN, "steps": DATA_STEPS}
    
    SNR = 0.5
    
    NDIM = 3
    NWALKERS = 100
    NUM_ITERATIONS = 12000
    
    
    # ------------------------------------------
    real_wave = evaluate_wave_fcn(WAVE_FCN, REAL_THETA, WAVE_KWARGS)
    noise = generate_noise(real_wave, snr=SNR)
    yerr = 1.0
                
    # Keyword args for the emcee_funcs.log_probability method
    log_prob_kwargs = {
        "noise": noise,
        "yerr": yerr,
        "ranges": RANGES,
        "wave_fcn": WAVE_FCN,
        "wave_kwargs": WAVE_KWARGS
    }
    
    # Prior probabilities for the parameters for each walker
    priors = np.random.uniform(low=RANGES[:,0], high=RANGES[:,1], size=(NWALKERS, NDIM))

    processor_pool = multiprocessing.Pool()
    
    sampler = emcee.EnsembleSampler(
        nwalkers=NWALKERS,
        ndim=NDIM,
        log_prob_fn=log_likelihood,
        kwargs=log_prob_kwargs,
        pool=processor_pool
    )
    
    start = time.time()
    samples = run_for_samples(sampler, priors=priors, num_iterations=NUM_ITERATIONS)
    end = time.time()
    
    rms = compare_for_error(real_theta=REAL_THETA, samples=samples, wave_fcn=WAVE_FCN, wave_kwargs=WAVE_KWARGS)
    
    real_likelihood = gaussian_likelihood(theta=REAL_THETA, noise=noise, yerr=yerr, wave_fcn=WAVE_FCN, wave_kwargs=WAVE_KWARGS)
    mean_theta_likelihood = gaussian_likelihood(theta=np.mean(samples, axis=0), noise=noise, yerr=yerr, wave_fcn=WAVE_FCN, wave_kwargs=WAVE_KWARGS)
    median_theta_likelihood = gaussian_likelihood(theta=np.median(samples, axis=0), noise=noise, yerr=yerr, wave_fcn=WAVE_FCN, wave_kwargs=WAVE_KWARGS)

    print(f"Finished processing MCMC in {end - start:.2f} seconds.\nError (RMS): {rms}\nLikelihoods (real, mean, median): {real_likelihood:.2f}, {mean_theta_likelihood:.2f}, {median_theta_likelihood:.2f}\nSamples count: {samples.shape}")
    
    # ------------- Plotting / Saving ------------------
    save = False
    if len(sys.argv) == 2:
        if sys.argv[1] == "save":
            save = True
    
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
    if save == True:
        save_figure(figure, "DampedSine/emcee/emcee_MainResults.png")
    else:
        plt.show()


