import emcee
import numpy as np
import matplotlib.pyplot as plt
from wave_funcs import emcee_wave
from utility_funcs import generate_noise
from plotting_funcs import plot_mcmc_wave_results
from emcee_funcs import *

    
if __name__ == "__main__":
    # -------------------------------------
    REAL_AMPLITUDE = 7.0
    REAL_DAMPING = 0.3
    REAL_ANGULAR_FREQ = 1.2
    DATA_STEPS = 1000
    TIMESPAN = 20
    NOISE_AMPLITUDE = 2.0
    NOISE_SCALE = 2.0
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
    time, real_wave = emcee_wave([REAL_AMPLITUDE, REAL_DAMPING, REAL_ANGULAR_FREQ], return_time=True, **WAVE_KWARGS)
    noise = generate_noise(real_wave, NOISE_SCALE, NOISE_AMPLITUDE)
    yerr =  NOISE_AMPLITUDE/REAL_AMPLITUDE
    lnprob_kwargs = {
        "noise": noise,
        "yerr": yerr,
        "ranges": RANGES,
        "wave_kwargs": WAVE_KWARGS
    }
    
    # Prior probabilities for the parameters for each walker
    priors = np.random.uniform(low=RANGES[:,0], high=RANGES[:,1], size=(NWALKERS, NDIM))

    metropolis_hastings_move = emcee.moves.MHMove(proposal_function=met_hastings_proposal)
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, log_prob_fn=log_probability, kwargs=lnprob_kwargs, moves=[(metropolis_hastings_move, 1.0)])
    
    # Burn in
    state = sampler.run_mcmc(priors, 300)
    sampler.reset()
    
    # Actual run
    position, prob, state = sampler.run_mcmc(state, NUM_ITERATIONS)
    
    print("priors:", priors, "state:", state, "position:", position, "prob:", prob, sep="\n"*2)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    # print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))
    
    samples = sampler.get_chain(flat=True)
    
    # ------------- Plotting ------------------
    figure, axes = plot_mcmc_wave_results(samples, real_wave, real_params=np.array([REAL_AMPLITUDE, REAL_DAMPING, REAL_ANGULAR_FREQ]), param_ranges=RANGES, wave_kwargs=WAVE_KWARGS, noise=noise, x=time)
    plt.show()


