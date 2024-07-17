import sys
import emcee
import numpy as np
import matplotlib.pyplot as plt
from wave_funcs import emcee_sine_gaussian_wave
from utility_funcs import guess_params, generate_noise, save_figure
from plotting_funcs import plot_sample_distributions, plot_real_vs_generated, plot_signal_in_noise


def met_hastings_proposal(coords: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray]:
    """
    Updates each walker with a new position vector (parameters) and a ratio of the log probability of
    the new position to the old position of each.

    Args:
        coords (np.ndarray): The positions of the walkers, shape = (n-walkers, n-dimensions).
        rng (np.random.Generator): Used for guessing new parameters.

    Returns:
        tuple[np.ndarray]: New position vector for each walker (n-walkers, n-dimensions) and log ratios (n-walkers,).
    """
    new_coords = np.empty_like(coords)      # new position vector for each walker
    log_ratios = np.empty(coords.shape[0])  # a ratio for each walker
    for i in range(coords.shape[0]):        # for each walker
        old_likelihood = gaussian(coords[i], noise, yerr, SG_KWARGS)
        # propose new parameters
        theta = guess_params(RANGES, rng)
        new_likelihood = gaussian(theta, noise, yerr, SG_KWARGS)
        # log ratio of the new likelihood to the old likelihood
        log_ratios[i] = new_likelihood / old_likelihood
        # new position vector in the parameter space
        new_coords[i] = theta
    return new_coords, log_ratios


def gaussian(theta: np.ndarray, noise: np.ndarray, yerr: float, wave_kwargs: dict) -> np.float64:
    return -0.5 * np.sum(((noise - emcee_sine_gaussian_wave(theta, **wave_kwargs))/yerr) ** 2)


def log_prior(theta: np.ndarray, ranges: np.ndarray):
    if np.all(ranges[:,0] < theta) and np.all(theta < ranges[:,1]):
        return 0.0
    return -np.inf


def log_probability(theta: np.ndarray, noise: np.ndarray, yerr: float, ranges: np.ndarray, wave_kwargs: dict) -> np.float64:
    lp = log_prior(theta, ranges)
    if not np.isfinite(lp):
        return -np.inf
    return lp + gaussian(theta, noise, yerr, wave_kwargs)

    
if __name__ == "__main__":
    # Real parameters for the signal
    REAL_AMPLITUDE = 3.0
    REAL_ANGULAR_FREQ = 10.0
    REAL_MEAN = 10.0
    REAL_DEVIATION = 3.3
    REAL_THETA = np.array(
        [REAL_AMPLITUDE, REAL_ANGULAR_FREQ, REAL_MEAN, REAL_DEVIATION]
    )
    # Parameter guessing ranges
    RANGES = np.array(
        [
            [0.1, 10.0],    # amplitude (A) range
            [0.1, 15.0],     # angular frequency (omega) range
            [0.1, 20.0],    # mean (m) range
            [0.1, 15.0]     # deviation (s) range
        ]
    )
    
    # Additional arguments to the sine-gaussian function
    DATA_STEPS = 1000   # data points (i.e length of x array)
    TIMESPAN = 20       # number of seconds
    SG_KWARGS = {"seconds": TIMESPAN, "steps": DATA_STEPS}
    
    # Noise magnitude
    NOISE_AMPLITUDE = 4.0
    NOISE_SCALE = 1.0
    
    # emcee parameters
    NDIM = 4
    NWALKERS = 50
    NUM_ITERATIONS = 10000
    #---------------------------------------------------------------------------------
    
    time, real_wave = emcee_sine_gaussian_wave(REAL_THETA, return_time=True, **SG_KWARGS)
    noise = generate_noise(real_wave, NOISE_SCALE, NOISE_AMPLITUDE)
    
    yerr =  NOISE_AMPLITUDE/REAL_AMPLITUDE
    lnprob_kwargs = {
        "noise": noise,
        "yerr": yerr,
        "ranges": RANGES,
        "wave_kwargs": SG_KWARGS
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
    
    # print("priors:", priors, "state:", state, "position:", position, "prob:", prob, sep="\n"*2)
    # print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    # print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))
    
    samples = sampler.get_chain(flat=True)
    
    # ------------- Plotting / Saving ------------------
    save = False
    if len(sys.argv) == 2:
        if sys.argv[1] == "save":
            save = True
            
    fig, axes = plot_sample_distributions(
        samples=samples, 
        real_theta=REAL_THETA, 
        xlabels=["Amplitude", "Angular Frequency", "Mean", "Deviation"], 
        ranges=RANGES
    )
    if save:
        save_figure(fig, "SineGaussian/SampleDistributions.png")
    else:
        plt.show()
    
    annotation = f"Real Parameters:\n$A = ${REAL_AMPLITUDE}\n$\\omega = ${REAL_ANGULAR_FREQ}\n$\\mu = ${REAL_MEAN}\n$\\sigma = ${REAL_DEVIATION}"
    fig, axes = plot_real_vs_generated(
        samples=samples, 
        real_theta=REAL_THETA, 
        x=time, 
        xlabel="time", 
        wave_kwargs=SG_KWARGS, 
        wave_fcn=emcee_sine_gaussian_wave,
        annotation=annotation
        )
    if save:
        save_figure(fig, "SineGaussian/RealVsGenerated.png")
    else:
        plt.show()
    
    fig, axes = plot_signal_in_noise(
        noise=noise,
        real_theta=REAL_THETA,
        x=time,
        xlabel="time",
        wave_kwargs=SG_KWARGS,
        wave_fcn=emcee_sine_gaussian_wave
    )
    if save:
        save_figure(fig, "SineGaussian/MaskedInNoise.png")
    else:
        plt.show()


