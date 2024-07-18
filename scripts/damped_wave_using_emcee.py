import emcee
import numpy as np
import matplotlib.pyplot as plt
from parallel_trials import emcee_trial
from wave_funcs import emcee_wave
from utility_funcs import generate_noise, save_figure
from plotting_funcs import PlottingWrapper
from emcee_funcs import *
import sys

    
if __name__ == "__main__":
    # -------------------------------------
    REAL_AMPLITUDE = 7.0
    REAL_DAMPING = 0.3
    REAL_ANGULAR_FREQ = 1.2
    REAL_THETA = np.array(
        [REAL_AMPLITUDE, REAL_DAMPING, REAL_ANGULAR_FREQ]
    )
    DATA_STEPS = 1000
    TIMESPAN = 20
    NOISE_AMPLITUDE = 2.0
    NOISE_SCALE = 1.5
    NDIM = 3
    NWALKERS = 50
    NUM_ITERATIONS = 10000
    RANGES = np.array(
        [
            [0.1, 10.0],    # amplitude (A) range
            [0.1, 3.0],     # damping (b) range
            [0.1, 10.0],    # angular frequency (omega) range
        ]
    )
    WAVE_KWARGS = {"phase": 0.0, "seconds": TIMESPAN, "steps": DATA_STEPS}
    # ------------------------------------------
    time, real_wave = emcee_wave(REAL_THETA, return_time=True, **WAVE_KWARGS)
    noise = generate_noise(real_wave, NOISE_SCALE, NOISE_AMPLITUDE)
    yerr =  NOISE_AMPLITUDE/REAL_AMPLITUDE
                
    # Keyword args for the emcee_funcs.log_probability method
    lnprob_kwargs = {
        "noise": noise,
        "yerr": yerr,
        "ranges": RANGES,
        "wave_fcn": emcee_wave,
        "wave_kwargs": WAVE_KWARGS
    }
    # Keyword args for the emcee_funcs.met_hastings_proposal method
    mh_kwargs = {
        "ranges": RANGES,
        "noise": noise,
        "yerr": yerr,
        "wave_fcn": emcee_wave,
        "wave_kwargs": WAVE_KWARGS
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
        wave_fcn=emcee_wave,
        wave_kwargs=WAVE_KWARGS
    )
    
    # ------------- Plotting / Saving ------------------
    save = False
    if len(sys.argv) == 2:
        if sys.argv[1] == "save":
            save = True
    
    plotter = PlottingWrapper(
        samples=samples,
        param_ranges=RANGES,
        wave_fcn=emcee_wave,
        wave_kwargs=WAVE_KWARGS,
        real_parameters=REAL_THETA,
        noise=noise
    )
    figure, axes = plotter.plot_mcmc_wave_results()
    if save == True:
        save_figure(figure, "DampedSine/emcee/emcee_MainResults.png")
    else:
        plt.show()


