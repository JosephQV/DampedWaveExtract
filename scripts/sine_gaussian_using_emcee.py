import sys
import emcee
import numpy as np
import matplotlib.pyplot as plt
from wave_funcs import emcee_sine_gaussian_wave
from utility_funcs import generate_noise, save_figure, evaluate_wave_fcn
from plotting_funcs import PlottingWrapper
from emcee_funcs import *
from parallel_trials import emcee_trial

    
if __name__ == "__main__":
    # Real parameters for the signal
    REAL_AMPLITUDE = 3.0
    REAL_ANGULAR_FREQ = 9.5
    REAL_MEAN = 10.0
    REAL_DEVIATION = 2.5
    REAL_THETA = np.array(
        [REAL_AMPLITUDE, REAL_ANGULAR_FREQ, REAL_MEAN, REAL_DEVIATION]
    )
    # Parameter guessing ranges
    RANGES = np.array(
        [
            [0.1, 10.0],    # amplitude (A) range
            [0.1, 15.0],    # angular frequency (omega) range
            [0.1, 20.0],    # mean (mu) range
            [0.1, 15.0]     # deviation (sigma) range
        ]
    )
    
    # Additional arguments to the sine-gaussian function
    DATA_STEPS = 1000   # data points (i.e length of x array)
    TIMESPAN = 20       # number of seconds
    SG_KWARGS = {"seconds": TIMESPAN, "steps": DATA_STEPS}
    
    # Noise magnitude
    NOISE_AMPLITUDE = 2.0
    NOISE_SCALE = 1.0
    
    # emcee parameters
    NDIM = 4
    NWALKERS = 50
    NUM_ITERATIONS = 10000
    #---------------------------------------------------------------------------------
    
    time, real_wave = emcee_sine_gaussian_wave(REAL_THETA, return_time=True, **SG_KWARGS)
    noise = generate_noise(real_wave, NOISE_SCALE, NOISE_AMPLITUDE)
    
    yerr =  NOISE_AMPLITUDE/REAL_AMPLITUDE
    
    # Keyword args for the emcee_funcs.log_probability method
    lnprob_kwargs = {
        "noise": noise,
        "yerr": yerr,
        "ranges": RANGES,
        "wave_fcn": emcee_sine_gaussian_wave,
        "wave_kwargs": SG_KWARGS
    }
    # Keyword args for the emcee_funcs.met_hastings_proposal method
    mh_kwargs = {
        "ranges": RANGES,
        "noise": noise,
        "yerr": yerr,
        "wave_fcn": emcee_sine_gaussian_wave,
        "wave_kwargs": SG_KWARGS
    }
    
    # Prior probabilities for the parameters for each walker
    priors = np.random.uniform(low=RANGES[:,0], high=RANGES[:,1], size=(NWALKERS, NDIM))
    metropolis_hastings_move = emcee.moves.MHMove(proposal_function=met_hastings_proposal)

    sampler_kwargs = {
        "nwalkers": NWALKERS,
        "ndim": NDIM,
        "log_prob_fn": log_probability,
        "kwargs": lnprob_kwargs,
        # "moves": [(metropolis_hastings_move, 1.0)]
    }
    
    samples, rms = emcee_trial(
        real=real_wave,
        num_iterations=NUM_ITERATIONS,
        sampler_kwargs=sampler_kwargs,
        priors=priors,
        wave_fcn=emcee_sine_gaussian_wave,
        wave_kwargs=SG_KWARGS
    )
    
    # ------------- Plotting / Saving ------------------
    save = False
    if len(sys.argv) == 2:
        if sys.argv[1] == "save":
            save = True
    
    plotter = PlottingWrapper(
        samples=samples,
        param_ranges=RANGES,
        wave_fcn=emcee_sine_gaussian_wave,
        wave_kwargs=SG_KWARGS,
        real_parameters=REAL_THETA,
        noise=noise
    )
            
    fig, axes = plotter.plot_sample_distributions(xlabels=["Amplitude", "Angular Frequency", "Mean", "Deviation"])
    if save:
        save_figure(fig, "SineGaussian/SampleDistributions.png")
    else:
        plt.show()
    
    annotation = f"Real Parameters:\n$A = ${REAL_AMPLITUDE}\n$\\omega = ${REAL_ANGULAR_FREQ}\n$\\mu = ${REAL_MEAN}\n$\\sigma = ${REAL_DEVIATION}"
    fig, axes = plotter.plot_real_vs_generated(annotation=annotation)
    if save:
        save_figure(fig, "SineGaussian/RealVsGenerated.png")
    else:
        plt.show()
    
    fig, axes = plotter.plot_signal_in_noise()
    if save:
        save_figure(fig, "SineGaussian/MaskedInNoise.png")
    else:
        plt.show()