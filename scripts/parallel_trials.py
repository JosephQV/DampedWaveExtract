import emcee
import numpy as np
from utility_funcs import compute_rms
from wave_funcs import *


def emcee_trial(
    real: np.ndarray,
    num_iterations: int,
    sampler_kwargs: dict,
    priors: np.ndarray,
    wave_fcn,
    wave_kwargs: dict
):
    sampler = emcee.EnsembleSampler(**sampler_kwargs)
    samples = run_for_samples(sampler, priors, num_iterations)
    
    rms = compare_for_error(real, samples, wave_fcn, wave_kwargs)
    
    return samples, rms


def run_for_samples(sampler: emcee.EnsembleSampler, priors: np.ndarray, num_iterations: int):
    state = sampler.run_mcmc(priors, 200)
    sampler.reset()
    sampler.run_mcmc(state, num_iterations)
    return sampler.get_chain(flat=True)


def compare_for_error(real: np.ndarray, samples: np.ndarray, wave_fcn, wave_kwargs: dict):
    mean_theta = np.mean(samples, axis=0)
    generated = eval(wave_fcn.__name__)(mean_theta, **wave_kwargs)
    return compute_rms(real, generated)



    