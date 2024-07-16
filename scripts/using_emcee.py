import emcee
import numpy as np
import matplotlib.pyplot as plt
from wave_funcs import emcee_wave
from utility_funcs import guess_params, generate_noise
from plotting_funcs import plot_mcmc_wave_results


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
        old_likelihood = gaussian(coords[i], noise, yerr, WAVE_KWARGS)
        # propose new parameters
        theta = guess_params(RANGES, rng)
        new_likelihood = gaussian(theta, noise, yerr, WAVE_KWARGS)
        # log ratio of the new likelihood to the old likelihood
        log_ratios[i] = new_likelihood / old_likelihood
        # new position vector in the parameter space
        new_coords[i] = theta
    return new_coords, log_ratios


def gaussian(theta: np.ndarray, noise: np.ndarray, yerr: float, wave_kwargs: dict) -> np.float64:
    return -0.5 * np.sum(((noise - emcee_wave(theta, **wave_kwargs))/yerr) ** 2)


def log_prior(theta: np.ndarray, ranges: np.ndarray):
    if np.all(ranges[:,0] < theta) and np.all(theta < ranges[:,1]):
        return 0.0
    return -np.inf


def log_probability(theta: np.ndarray, noise: np.ndarray, yerr: float, ranges: np.ndarray, wave_kwargs: dict) -> np.float64:
    lp = log_prior(theta, ranges)
    if not np.isfinite(lp):
        return -np.inf
    return lp + gaussian(theta, noise, yerr, wave_kwargs)


def cov(x, y):
    mx = np.mean(x)
    my = np.mean(y)
    gm = np.mean((x - mx) * (y - my))  
    return gm

    
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
    yerr = 0.10 * np.mean(noise)
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


