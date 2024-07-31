import os, sys, json
import matplotlib.pyplot as plt
import emcee
from plotting_funcs import *
from utility_funcs import generate_noise


if __name__ == "__main__":
    param_file = sys.argv[1]
    with open(param_file) as config:
        params = json.load(config)
        
    samples_file = sys.argv[2]
    
    outdir = sys.argv[3]
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
    names = []
    labels = []
    for p in parameters:
        real_theta.append(p["real_value"])
        ranges.append(p["range"])
        names.append(p["name"])
        labels.append(p["label"])
    real_theta = np.array(real_theta)
    ranges = np.array(ranges)
        
    real_wave = evaluate_wave_fcn(wave_fcn, real_theta, wave_kwargs)
    noise = generate_noise(real_wave, snr)
    
    reader = emcee.backends.HDFBackend(
        filename=samples_file,
        read_only=True
    )
    samples = reader.get_chain(flat=True, thin=thin_by, discard=int(niterations * burn_percentage))
    single_chain = reader.get_chain(flat=False)[:, np.random.randint(0, samples.shape[1])]
    
    plotter = PlottingWrapper(
        samples=samples,
        single_chain=single_chain,
        real_parameters=real_theta,
        param_ranges=ranges,
        param_names=names,
        param_labels=labels,
        wave_fcn=wave_fcn,
        wave_kwargs=wave_kwargs,
        snr=snr
    )
    
    fig, axes = plotter.plot_sample_distributions()
    fig.savefig(f"{outdir}/SampleDistributions.png")
    
    fig, axes = plotter.plot_real_vs_generated()
    fig.savefig(f"{outdir}/RealVsGenerated.png")
    
    fig, axes = plotter.plot_signal_in_noise()
    fig.savefig(f"{outdir}/MaskedInNoise.png")
    
    fig, axes = plotter.plot_posterior_traces(iter_start=0, iter_end=niterations)
    fig.savefig(f"{outdir}/SampleTraces.png")