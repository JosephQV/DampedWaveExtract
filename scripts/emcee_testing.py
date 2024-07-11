import emcee
import numpy as np
import matplotlib.pyplot as plt
from wave_func import mcmc_wave

def generate_noise(data, scale, noise_amp=1.0):
    noise = 2 * noise_amp * np.random.normal(loc=0.0, scale=scale, size=len(data)) - noise_amp
    return data + noise
    
def compute_rms(observed, predicted):
    diff = observed - predicted
    return np.sqrt(np.mean(diff**2))
    
def likelihood(params, **kwargs):
    fcn = kwargs["fcn"]
    noise = kwargs["noise"]
    fcn_kwargs = kwargs["fcn_kwargs"]
    tol = kwargs["tol"]
    
    trial_vals = eval(fcn.__name__)(*params, **fcn_kwargs)
    rms_error = compute_rms(noise, trial_vals)
    likelihood = 1 / (rms_error + tol)
    return likelihood

if __name__ == "__main__":
    REAL_AMPLITUDE = 7.0
    REAL_DAMPING = 0.3
    REAL_ANGULAR_FREQ = 1.2
    DATA_STEPS = 10000
    TIMESPAN = 20
    NOISE_AMPLITUDE = 0.0
    NOISE_SCALE = 0.0
    NDIM = 3
    NWALKERS = 32
    NUM_ITERATIONS = 10000
    
    wave_kwargs = {"phase": 0.0, "seconds": TIMESPAN, "steps": DATA_STEPS}
    time, real_wave = mcmc_wave(REAL_AMPLITUDE, REAL_DAMPING, REAL_ANGULAR_FREQ, return_time=True, **wave_kwargs)
    noise = generate_noise(real_wave, NOISE_SCALE, NOISE_AMPLITUDE)

    likelihood_kwargs = {
        "fcn": mcmc_wave,
        "noise": noise,
        "fcn_kwargs": wave_kwargs,
        "tol": 1e-9
    }
    
    # Prior probabilities for the parameters for each walker
    p0 = np.random.rand(NWALKERS, NDIM)
    
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, likelihood, kwargs=likelihood_kwargs)
    # Burn in
    state = sampler.run_mcmc(p0, 100)
    sampler.reset()

    sampler.run_mcmc(state, NUM_ITERATIONS)
    samples = sampler.get_chain(flat=True)

    print("p0:", p0, "state:", state, "samples:", samples, samples.shape, sep="\n"*2)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))
    
    amplitude_samples = samples[:, 0]
    damping_samples = samples[:, 1]
    angular_freq_samples = samples[:, 2]

    mean_amplitude = np.mean(amplitude_samples)
    mean_damping = np.mean(damping_samples)
    mean_angular_freq = np.mean(angular_freq_samples)

    std_amplitude = np.std(amplitude_samples)
    std_damping = np.std(damping_samples)
    std_angular_freq = np.std(angular_freq_samples)

    result_wave = mcmc_wave(mean_amplitude, mean_damping, mean_angular_freq, **wave_kwargs)
    figure = plt.figure(figsize=(13, 7), layout="constrained")
    figure.suptitle("(emcee) Extracting Signal Parameters From Noise")
    axes = figure.subplot_mosaic(
    """
    WWN.
    ABF.
    """,
    width_ratios=[1, 1, 1, 0.05]
    )

    axes["A"].hist(amplitude_samples, bins=100, density=True, label="estimated distribution")
    axes["A"].set_xlabel("Amplitude")
    ylim = axes["A"].get_ylim()
    axes["A"].plot([REAL_AMPLITUDE, REAL_AMPLITUDE], [0.0, ylim[1]], label="true value")
    axes["A"].plot([mean_amplitude, mean_amplitude], [0.0, ylim[1]], label="sample mean")
    axes["A"].plot([mean_amplitude+std_amplitude, mean_amplitude+std_amplitude], [0.0, ylim[1]], "r--", label="+1 std")
    axes["A"].plot([mean_amplitude-std_amplitude, mean_amplitude-std_amplitude], [0.0, ylim[1]], "r--", label="-1 std")
    axes["A"].legend(prop={"size": 8})
    # axes["A"].set_xlim(left=ranges[0, 0], right=ranges[0, 1])


    axes["B"].hist(damping_samples, bins=100, density=True, label="estimated distribution")
    axes["B"].set_xlabel("Damping")
    ylim = axes["B"].get_ylim()
    axes["B"].plot([REAL_DAMPING, REAL_DAMPING], [0.0, ylim[1]], label = "true value")
    axes["B"].plot([mean_damping, mean_damping], [0.0, ylim[1]], label="sample mean")
    axes["B"].plot([mean_damping+std_damping, mean_damping+std_damping], [0.0, ylim[1]], "r--", label="+1 std")
    axes["B"].plot([mean_damping-std_damping, mean_damping-std_damping], [0.0, ylim[1]], "r--", label="-1 std")
    axes["B"].legend(prop={"size": 8})
    # axes["B"].set_xlim(left=ranges[1, 0], right=ranges[1, 1])

    axes["F"].hist(angular_freq_samples, bins=100, density=True, label="estimated distribution")
    axes["F"].set_xlabel("Angular frequency")
    ylim = axes["F"].get_ylim()
    axes["F"].plot([REAL_ANGULAR_FREQ, REAL_ANGULAR_FREQ], [0.0, ylim[1]], label = "true value")
    axes["F"].plot([mean_angular_freq, mean_angular_freq], [0.0, ylim[1]], label="sample mean")
    axes["F"].plot([mean_angular_freq+std_angular_freq, mean_angular_freq+std_angular_freq], [0.0, ylim[1]], "r--", label="+1 std")
    axes["F"].plot([mean_angular_freq-std_angular_freq, mean_angular_freq-std_angular_freq], [0.0, ylim[1]], "r--", label="-1 std")
    axes["F"].legend(prop={"size": 8})
    # axes["F"].set_xlim(left=ranges[2, 0], right=ranges[2, 1])
        
    axes["W"].plot(time, real_wave, "r", label="real")
    axes["W"].set_xlabel("time")
    rms_deviation = compute_rms(real_wave, result_wave)
    axes["W"].annotate(f"Real Parameters:\nAmplitude: {REAL_AMPLITUDE}\nDamping: {REAL_DAMPING}\nAngular Frequency: {REAL_ANGULAR_FREQ}\n\nRMS Deviation: {rms_deviation:.3f}", xy=(0.75, 0.6), xycoords="axes fraction")
    axes["W"].plot(time, result_wave, "g", label="generated")
    axes["W"].legend()

    axes["N"].plot(time, noise)
    axes["N"].set_xlabel("time")
    axes["N"].annotate(f"Noise amplitude: {NOISE_AMPLITUDE}", xy=(0.53, 0.9), xycoords="axes fraction")

    plt.show()


