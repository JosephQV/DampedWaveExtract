import numpy as np
import bilby
from wave_funcs import *
from utility_funcs import generate_noise, evaluate_wave_fcn
import json, sys, os


if __name__ == "__main__":
    param_file = sys.argv[1]
    with open(param_file) as config:
        params = json.load(config)
    
    outdir = sys.argv[2]
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    parameters = params["parameters"]
    snr = params["SNR"]
    yerr = params["y-error"]
    ndim = params["ndim"]
    nwalkers = params["nwalkers"]
    niterations = params["niterations"]
    burn_percentage = params["burn_percentage"]
    thin_by = params["thin_by"]
    timespan = params["x_seconds"]
    
    # ----------------------------------------------------------------------------------------------------
    
    real_theta = []
    priors = bilby.core.prior.PriorDict()
    for p in parameters:
        real_theta.append(p["real_value"])
        prior = bilby.core.prior.Uniform(minimum=p["range"][0], maximum=p["range"][1], name=p["name"], latex_label=p["label"])
        priors[p["arg_name"]] = prior
    
    real_theta = np.array(real_theta)
    
    time = np.linspace(0, timespan, 1000)
    real_wave = bilby_sine_gaussian_wave(time, *real_theta)
    noise = generate_noise(real_wave, snr)
    
    sampler = "bilby_mcmc"
    sampler_kwargs = {
        "nsamples": niterations,
        "ntemps": nwalkers,
        "thin_by_nact": 1.0 / thin_by
    }
    
    likelihood = bilby.core.likelihood.GaussianLikelihood(
        x=time, 
        y=noise, 
        func=bilby_sine_gaussian_wave,
        sigma=yerr 
    )
    
    result = bilby.core.sampler.run_sampler(
        likelihood=likelihood,
        priors=priors,
        label=f"{sampler}\\{bilby_sine_gaussian_wave.__name__}",
        outdir=outdir,
        injection_parameters=real_theta,
        plot=True,
        npool=64,
        sampler=sampler,
        **sampler_kwargs
    )
    result.plot_with_data(
        model=bilby_sine_gaussian_wave,
        x=time,
        y=noise,
    )