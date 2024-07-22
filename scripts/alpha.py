from multiprocessing.pool import Pool
import time, csv
import numpy as np
from utility_funcs import generate_noise, evaluate_wave_fcn
from wave_funcs import *
from emcee_funcs import *
import emcee
import pathlib


def initialize_shared_config():
    real_amplitude = 4.0
    real_angular_freq = 8.5
    real_mean = 10.0
    real_deviation = 3.0
    real_theta = np.array(
        [real_amplitude, real_angular_freq, real_mean, real_deviation]
    )
    # Additional arguments to the sine-gaussian function
    data_steps = 1000    # data points (i.e length of x array)
    timespan = 30        # number of seconds
    wave_kwargs = {"seconds": timespan, "steps": data_steps}
    wave_fcn = sine_gaussian_wave
    real_wave = evaluate_wave_fcn(wave_fcn, real_theta, wave_kwargs)

    ranges = np.array(
        [
            [0.1, 20.0],    # amplitude (A) range
            [0.1, 20.0],    # angular frequency (omega) range
            [0.1, timespan],    # mean (mu) range
            [0.1, 20.0]     # deviation (sigma) range
        ]
    )
    
    yerr = 1.0
    # emcee parameters
    ndim = 4
    nwalkers = 100
    num_iterations = 10000
    
    shared_config = {
        "real_theta": real_theta,
        "real_wave": real_wave,
        "wave_fcn": wave_fcn,
        "wave_kwargs": wave_kwargs,
        "real_wave": real_wave,
        "ranges": ranges,
        "yerr": yerr,
        "ndim": ndim,
        "nwalkers": nwalkers,
        "num_iterations": num_iterations
    }
    return shared_config


def initialize_process_sampler(shared_config, snr):
    real_wave = shared_config["real_wave"]
    ndim = shared_config["ndim"]
    nwalkers = shared_config["nwalkers"]
    ranges = shared_config["ranges"]
    yerr = shared_config["yerr"]
    wave_fcn = shared_config["wave_fcn"]
    wave_kwargs = shared_config["wave_kwargs"]
        
    noise = generate_noise(real_wave, snr)
    
    # initialize_metropolis_hastings_variables(
    #     ranges=ranges,
    #     noise=noise,
    #     yerr=yerr,
    #     wave_fcn=wave_fcn,
    #     wave_kwargs=wave_kwargs
    # )
    # metropolis_hastings_move = emcee.moves.MHMove(proposal_function=met_hastings_proposal)
    
    # Keyword args for the emcee_funcs.log_probability method
    log_prob_kwargs = {
        "noise": noise,
        "yerr": yerr,
        "ranges": ranges,
        "wave_fcn": wave_fcn,
        "wave_kwargs": wave_kwargs
    }
    
    sampler = emcee.EnsembleSampler(
        nwalkers=nwalkers,
        ndim=ndim,
        log_prob_fn=log_likelihood,
        kwargs=log_prob_kwargs,
        # moves=[(metropolis_hastings_move, 1.0)]
    )
    
    priors = np.random.uniform(low=ranges[:,0], high=ranges[:,1], size=(nwalkers, ndim))

    return sampler, priors
    

def process_for_samples(snr):
    shared_config = initialize_shared_config()
    sampler, priors = initialize_process_sampler(shared_config, snr)
    
    samples = run_for_samples(sampler, priors, num_iterations=shared_config["num_iterations"])
    rms = compare_for_error(real_theta=shared_config["real_theta"], samples=samples, wave_fcn=shared_config["wave_fcn"], wave_kwargs=shared_config["wave_kwargs"])
    return np.median(samples, axis=0), rms, snr


def run_in_parallel(snrs: np.ndarray):
    with Pool() as pool:
        results = pool.map(process_for_samples, snrs)
    
    return results  # [(median1, rms1, snr1), (median1, rms2, snr2), ...]
        
    
def write_to_csv(output_file, results):
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["SNR", "RMS"])
        for snr, rms in results:
            writer.writerow([snr, rms])
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    snrs = np.linspace(1.00, 0.10, num=25)

    start = time.time()
    results = run_in_parallel(snrs)
    t = time.time() - start
    print(f"Finished. Time: {t}")
    
    f_result = []
    for median_theta, rms, snr in results:
        print(f"SNR: {snr:.2f}, RMS: {rms:.3f}")
        f_result.append((snr, rms))
    
    fig = plt.figure()
    axes = fig.add_axes()
    axes.scatter(f_result[0], f_result[1], "red")
    axes.set_xlabel("SNR")
    axes.set_ylabel("Error (RMS)")
    fig.suptitle("Error by Signal to Noise Ratio")
    
    output_file = pathlib.Path.cwd().joinpath("ErrorBySNR.csv")
    write_to_csv(output_file, f_result)