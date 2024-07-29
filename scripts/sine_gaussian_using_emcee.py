import multiprocessing.pool
import sys, time
import numpy as np
import matplotlib.pyplot as plt
from wave_funcs import sine_gaussian_wave
from utility_funcs import generate_noise, save_figure
from plotting_funcs import PlottingWrapper
from emcee_funcs import *
import multiprocessing


if __name__ == "__main__":
    # Real parameters for the signal
    REAL_AMPLITUDE = 4.0
    REAL_ANGULAR_FREQ = 7.5
    REAL_MEAN = 20.0
    REAL_DEVIATION = 3.0
    REAL_THETA = np.array(
        [REAL_AMPLITUDE, REAL_ANGULAR_FREQ, REAL_MEAN, REAL_DEVIATION]
    )
    
    # Additional arguments to the sine-gaussian function
    DATA_STEPS = 1000    # data points (i.e length of x array)
    TIMESPAN = 30        # number of seconds
    WAVE_KWARGS = {"seconds": TIMESPAN, "steps": DATA_STEPS}
    WAVE_FCN = sine_gaussian_wave
    
    # Parameter guessing ranges
    RANGES = np.array(
        [
            [0.1, 20.0],    # amplitude (A) range
            [0.1, 20.0],    # angular frequency (omega) range
            [0.1, TIMESPAN],    # mean (mu) range
            [0.1, 20.0]     # deviation (sigma) range
        ]
    )
    
    # Signal to Noise Ratio ( greater than 1 is very little noise, close to 0 is intense noise)
    SNR = 0.23
    
    # emcee parameters
    NDIM = 4
    NWALKERS = 100
    NUM_ITERATIONS = 10000
    #---------------------------------------------------------------------------------
    
    real_wave = evaluate_wave_fcn(WAVE_FCN, REAL_THETA, WAVE_KWARGS)
    noise = generate_noise(real_wave, snr=SNR)
    
    yerr = 1.0  # sigma, deviation for guessing
    
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
    sampler.run_mcmc(priors, NUM_ITERATIONS)
    end = time.time()
    samples = sampler.get_chain(flat=True)
    
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
            
    fig, axes = plotter.plot_sample_distributions(xlabels=["Amplitude", "Angular Frequency", "Mean", "Deviation"])
    if save:
        save_figure(fig, "SineGaussian/test/SampleDistributions.png")
    else:
        plt.show()
    
    annotation = f"Real Parameters:\n$A = ${REAL_AMPLITUDE}\n$\\omega = ${REAL_ANGULAR_FREQ}\n$\\mu = ${REAL_MEAN}\n$\\sigma = ${REAL_DEVIATION}"
    fig, axes = plotter.plot_real_vs_generated(annotation=annotation)
    if save:
        save_figure(fig, "SineGaussian/test/RealVsGenerated.png")
    else:
        plt.show()
    
    fig, axes = plotter.plot_signal_in_noise()
    if save:
        save_figure(fig, "SineGaussian/test/MaskedInNoise.png")
    else:
        plt.show()
    
    plotter.samples = sampler.get_chain()[:, np.random.randint(0, NWALKERS)]   # using one of the chains from one random walker for this plot
    fig, axes = plotter.plot_posterior_traces(ylabels=["Amplitude", "Angular Frequency", "Mean", "Deviation"], iter_start=0, iter_end=NUM_ITERATIONS)
    if save:
        save_figure(fig, "SineGaussian/test/SampleTraces.png")
    else:
        plt.show()