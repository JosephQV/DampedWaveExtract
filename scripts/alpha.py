import multiprocessing
import time, csv
import numpy as np
from utility_funcs import generate_noise, evaluate_wave_fcn
from wave_funcs import *
from emcee_funcs import *
from parallel_emcee_trials import emcee_trial
import emcee


def initialize_shared_variables():
    global RANGES, REAL_WAVE, WAVE_FCN, WAVE_KWARGS, SAMPLER_KWARGS, NDIM, NWALKERS, NUM_ITERATIONS

    real_amplitude = 4.0
    real_angular_freq = 8.5
    real_mean = 10.0
    real_deviation = 3.0
    real_theta = np.array(
        [real_amplitude, real_angular_freq, real_mean, real_deviation]
    )
    
    # Additional arguments to the sine-gaussian function
    data_steps = 10000   # data points (i.e length of x array)
    timespan = 30        # number of seconds
    WAVE_KWARGS = {"seconds": timespan, "steps": data_steps}
    WAVE_FCN = sine_gaussian_wave
    REAL_WAVE = evaluate_wave_fcn(WAVE_FCN, real_theta, WAVE_KWARGS)

    RANGES = np.array(
        [
            [0.1, 10.0],    # amplitude (A) range
            [0.1, 15.0],    # angular frequency (omega) range
            [0.1, timespan],    # mean (mu) range
            [0.1, 15.0]     # deviation (sigma) range
        ]
    )
    
    # emcee parameters
    NDIM = 4
    NWALKERS = 20
    NUM_ITERATIONS = 100 
    
    
def initialize_process_noise(*noise_args):
    global NOISE
    
    noise_amplitude = noise_args[0]
    noise_scale = noise_args[1]
    
    NOISE = generate_noise(REAL_WAVE, noise_scale, noise_amplitude)


def initialize_process_sampler():
    global SAMPLER_KWARGS, PRIORS
    
    YERR =  0.05 * np.mean(NOISE)
    
    # Keyword args for the emcee_funcs.log_probability method
    lnprob_kwargs = {
        "noise": NOISE,
        "yerr": YERR,
        "ranges": RANGES,
        "wave_fcn": WAVE_FCN,
        "wave_kwargs": WAVE_KWARGS
    }
    
    # Prior probabilities for the parameters for each walker
    PRIORS = np.random.uniform(low=RANGES[:,0], high=RANGES[:,1], size=(NWALKERS, NDIM))

    SAMPLER_KWARGS = {
        "nwalkers": NWALKERS,
        "ndim": NDIM,
        "log_prob_fn": log_probability,
        "kwargs": lnprob_kwargs,
        "moves": [(metropolis_hastings_move, 1.0)]
    }
    
    RANGES = ranges
    NOISE = noise
    YERR = yerr
    WAVE_FCN = wave_fcn
    WAVE_KWARGS = wave_kwargs


def process_for_samples():
    samples, rms = emcee_trial(
        real=REAL_WAVE,
        num_iterations=NUM_ITERATIONS,
        sampler_kwargs=SAMPLER_KWARGS,
        priors=PRIORS,
        wave_fcn=WAVE_FCN,
        wave_kwargs=WAVE_KWARGS
    )


def run_in_parallel():
    pass
    # close the processes (exit the with block)
    

def write_to_csv():
    pass


if __name__ == "__main__":
    # Initialize some data store that holds the configuration for the trial, the wave parameters, etc.
    initialize_shared_variables()
    
    # Initialize some data store like a queue to hold the results of each trial
    
    
    # Initialize some array of values for noise strength (signal to noise ratios)
    snrs = np.linspace(1.00, 0.10, num=90)

    # run in parallel the processes that will map noise inputs to their trial of emcee
    start = time.time()
    error_by_noise = run_in_parallel()
    t = time.time() - start
    print(f"Finished. Time: {t}")
    
    # write the output to a csv file
    write_to_csv()
    
    pass