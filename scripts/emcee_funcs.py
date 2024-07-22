import numpy as np
import emcee
from utility_funcs import guess_params, evaluate_wave_fcn, compute_rms

# RANGES = None
# NOISE = None
# YERR = None
# WAVE_FCN = None
# WAVE_KWARGS = None

def met_hastings_proposal(coords: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray]:
    """
    Updates each walker with a new position vector (parameters) and a ratio of the log probability of
    the new position to the old position of each.
    **The parameter ranges (RANGES), noise data (NOISE), y-error (YERR), the wave function (WAVE_FCN), and wave keyword arguments (WAVE_KWARGS), must be initialized
    as global variables before passing this function to the emcee sampler in an MHMove.**

    Args:
        coords (np.ndarray): The positions of the walkers, shape = (n-walkers, n-dimensions).
        rng (np.random.Generator): Used for guessing new parameters.
        kwargs:
        param_ranges: Ranges for guessing parameters when making proposals.
        noise: The noisy data that proposed waves are compared to.
        yerr: 0.0 - 1.0, the percentage of uncertainty in the data.
        wave_fcn: The callable function that will be used for generating waves at each proposal.
        wave_kwargs: Keyword arguments to be passed to the wave_fcn for consistency.

    Returns:
        tuple[np.ndarray]: New position vector for each walker (n-walkers, n-dimensions) and log ratios (n-walkers,).
    """
    new_coords = np.empty_like(coords)      # new position vector for each walker
    log_ratios = np.empty(coords.shape[0])  # a ratio for each walker
    for i in range(coords.shape[0]):        # for each walker
        old_likelihood = gaussian_likelihood(coords[i], noise, yerr, wave_fcn, wave_kwargs)
        # propose new parameters
        theta = guess_params(ranges, rng)
        new_likelihood = gaussian_likelihood(theta, noise, yerr, wave_fcn, wave_kwargs)
        # log ratio of the new likelihood to the old likelihood
        log_ratios[i] = new_likelihood / old_likelihood
        # new position vector in the parameter space
        new_coords[i] = theta
    return new_coords, log_ratios


def gaussian_likelihood(theta: np.ndarray, noise: np.ndarray, yerr: float, wave_fcn, wave_kwargs: dict) -> np.float64:
    wave = evaluate_wave_fcn(wave_fcn, theta, wave_kwargs)
    return -0.5 * np.sum(((noise - wave)/yerr) ** 2)


def log_prior(theta: np.ndarray, ranges: np.ndarray):
    if np.all(ranges[:,0] < theta) and np.all(theta < ranges[:,1]):
        return 0.0
    return -np.inf


def log_likelihood(theta: np.ndarray, noise: np.ndarray, yerr: float, ranges: np.ndarray, wave_fcn, wave_kwargs: dict) -> np.float64:
    lp = log_prior(theta, ranges)
    if not np.isfinite(lp):
        return -np.inf
    return lp + gaussian_likelihood(theta, noise, yerr, wave_fcn, wave_kwargs)


def emcee_trial(
    real: np.ndarray,
    num_iterations: int,
    sampler_kwargs: dict,
    priors: np.ndarray,
    wave_fcn,
    wave_kwargs: dict,
    use_met_hastings: bool = False,
    met_hastings_kwargs: dict | None = None
):
    if use_met_hastings == True:
        initialize_metropolis_hastings_variables(**met_hastings_kwargs)
        mh = emcee.moves.MHMove(proposal_function=met_hastings_proposal)
        sampler = emcee.EnsembleSampler(moves=[(mh, 1.0)])
    else:
        sampler = emcee.EnsembleSampler(**sampler_kwargs)
    
    samples = run_for_samples(sampler, priors, num_iterations)
    
    rms = compare_for_error(real, samples, wave_fcn, wave_kwargs)
    
    return samples, rms


def run_for_samples(sampler: emcee.EnsembleSampler, priors: np.ndarray, num_iterations: int):
    state = sampler.run_mcmc(priors, 200)
    sampler.reset()
    sampler.run_mcmc(state, num_iterations)
    return sampler.get_chain(flat=True)


def compare_for_error(real_theta: np.ndarray, samples: np.ndarray, wave_fcn, wave_kwargs: dict):
    median_theta = np.median(samples, axis=0)
    real_wave = evaluate_wave_fcn(wave_fcn, real_theta, wave_kwargs)
    generated = evaluate_wave_fcn(wave_fcn, median_theta, wave_kwargs)
    return compute_rms(real_wave, generated)


def initialize_metropolis_hastings_variables(ranges: np.ndarray, noise: np.ndarray, yerr: float, wave_fcn, wave_kwargs: dict):
    global RANGES, NOISE, YERR, WAVE_FCN, WAVE_KWARGS
    
    RANGES = ranges
    NOISE = noise
    YERR = yerr
    WAVE_FCN = wave_fcn
    WAVE_KWARGS = wave_kwargs
    