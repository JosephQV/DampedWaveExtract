import sys, time, os, json
import numpy as np
from utility_funcs import generate_noise
from emcee_funcs import *
import multiprocessing


if __name__ == "__main__":
    param_file = sys.argv[1]
    with open(param_file) as config:
        params = json.load(config)
    
    outdir = sys.argv[2]
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    wave_fcn = params["wave_function"]
    wave_kwargs = params["wave_kwargs"]
    parameters = params["parameters"]
    snr = params["SNR"]
    yerr = params["y-error"]
    ndim = params["ndim"]
    nwalkers = params["nwalkers"]
    niterations = params["niterations"]
    burn_percentage = params["burn_percentage"]
    thin_by = params["thin_by"]

    #---------------------------------------------------------------------------------
    
    real_theta = []
    ranges = []
    for p in parameters:
        real_theta.append(p["real_value"])
        ranges.append(p["range"])
    real_theta = np.array(real_theta)
    ranges = np.array(ranges)
        
    real_wave = evaluate_wave_fcn(wave_fcn, real_theta, wave_kwargs)
    noise = generate_noise(real_wave, snr)
    
    # Keyword args for the emcee_funcs.log_probability method
    log_prob_kwargs = {
        "noise": noise,
        "yerr": yerr,
        "ranges": ranges,
        "wave_fcn": wave_fcn,
        "wave_kwargs": wave_kwargs
    }
    
    # Prior probabilities for the parameters for each walker
    priors = np.random.uniform(low=ranges[:,0], high=ranges[:,1], size=(nwalkers, ndim))

    processor_pool = multiprocessing.Pool(64)
    
    backend = emcee.backends.HDFBackend(
        filename=f"{outdir}/sample_chain.hdf5"
    )
    backend.reset(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(
        nwalkers=nwalkers,
        ndim=ndim,
        log_prob_fn=log_likelihood,
        kwargs=log_prob_kwargs,
        pool=processor_pool,
        backend=backend
    )
    
    start = time.time()
    sampler.run_mcmc(
        initial_state=priors,
        nsteps=niterations,
        store=True
    )
    end = time.time()
    samples = sampler.get_chain(flat=True, thin=thin_by, discard=int(niterations * burn_percentage))
        
    real_likelihood = gaussian_likelihood(theta=real_theta, noise=noise, yerr=yerr, wave_fcn=wave_fcn, wave_kwargs=wave_kwargs)
    mean_theta_likelihood = gaussian_likelihood(theta=np.mean(samples, axis=0), noise=noise, yerr=yerr, wave_fcn=wave_fcn, wave_kwargs=wave_kwargs)
    median_theta_likelihood = gaussian_likelihood(theta=np.median(samples, axis=0), noise=noise, yerr=yerr, wave_fcn=wave_fcn, wave_kwargs=wave_kwargs)
    
    best_theta, best_likelihood = get_best_theta(sampler)
    
    rms_median = compute_rms(
        observed=evaluate_wave_fcn(wave_fcn, real_theta, wave_kwargs),
        predicted=evaluate_wave_fcn(wave_fcn, np.median(samples, axis=0), wave_kwargs)
    )
    rms_best = compute_rms(
        observed=evaluate_wave_fcn(wave_fcn, real_theta, wave_kwargs),
        predicted=evaluate_wave_fcn(wave_fcn, best_theta, wave_kwargs)
    )
    
    try:
        no_thin_autocorr = sampler.get_autocorr_time()
        thinned_autocorr = sampler.get_autocorr_time(thin=thin_by, discard=burn_percentage)
    except Exception as e:
        no_thin_autocorr = "failed"
        thinned_autocorr = "failed"
        print(f"Failed to retrieve autocorrelation time estimate: {str(e)}")
    
    params["samples size"] = list(samples.shape)
    params["best likelihood"] = float(best_likelihood)
    params["best theta"] = list(best_theta)
    params["autocorrelation"] = thinned_autocorr
    params["rms"] = rms_best
    
    try:
        with open(f"{outdir}/run_config.json", mode="x") as out_config:
            json.dump(params, out_config)
    except Exception as e:
        print(f"Failed to write run config to file: {str(e)}")
    
    print(f"Finished processing MCMC in {end - start:.2f} seconds.\nSamples Size: {samples.shape}\tBurn-in Percentage: {burn_percentage*100:.1f}%\tThinned By: {thin_by}")
    print(f"Likelihoods (real, mean, median thetas): {real_likelihood:.2f}, {mean_theta_likelihood:.2f}, {median_theta_likelihood:.2f}")
    print(f"Max Likelihood(s): {best_likelihood}")
    print(f"Best Theta(s): {best_theta}")
    print(f"RMS Error (Median - Real, Best - Real): {rms_median}, {rms_best}")
    print(f"Autocorrelation Time (no thinning, with thinning): {no_thin_autocorr}, {thinned_autocorr}")