import emcee
import numpy as np
import matplotlib.pyplot as plt

def emcee_wave(theta, phase=0.0, seconds=30, steps=1000, return_time=False):
    amplitude, damping, angular_freq = theta
    time = np.linspace(0, seconds, steps)
    displacement = amplitude * np.exp(-1.0 * damping / 2.0 * time) * np.sin((angular_freq * time) + phase)
    if return_time:
        return time, displacement
    return displacement

def generate_noise(data: np.ndarray, scale: float, noise_amp: float = 1.0) -> np.ndarray:
    noise = 2 * noise_amp * np.random.normal(loc=0.0, scale=scale, size=len(data)) - noise_amp
    return data + noise

def guess_params(rng: np.random.Generator, param_ranges: np.ndarray) -> np.ndarray:
    return rng.uniform(low=param_ranges[:,0], high=param_ranges[:,1], size=len(param_ranges))

def compute_rms(observed: np.ndarray, predicted: np.ndarray) -> float:
    not_nans = np.where(np.isnan(predicted) == False)
    diff = observed[not_nans] - predicted[not_nans]
    return np.sqrt(np.mean(diff**2))

def likelihood(theta: np.ndarray, **kwargs) -> float:
    fcn = kwargs["fcn"]
    noise = kwargs["noise"]
    fcn_kwargs = kwargs["fcn_kwargs"]
    tol = kwargs["tol"]
    
    trial_vals = eval(fcn.__name__)(theta, **fcn_kwargs)
    rms_error = compute_rms(noise, trial_vals)
    likelihood = 1 / (rms_error + tol)
    #print("params:",theta,"kwargs:",kwargs,"rms:",rms_error,"likelihood:",likelihood,sep="\n")
    return likelihood

def mh_proposal(coords: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray]:
    old_likelihoods = np.empty_like(coords)
    new_likelihoods = np.empty_like(coords)
    
    # compute the likelihood of the current positions and of the proposed params (for each walker)
    for i in range(len(coords)):
        old_likelihoods[i] = likelihood(coords[i], **likelihood_kwargs)
        # propose random parameters (for each walker)
        theta = guess_params(rng, RANGES)
        new_likelihoods[i] = likelihood(theta, **likelihood_kwargs)
        coords[i] = theta
    
    # return proposed position and the log ratio of the new likelihood to the old likelihood
    return coords, np.mean(np.log(new_likelihoods) - np.log(old_likelihoods))

def lnlike(theta, noise, yerr, wave_kwargs):
    return -0.5 * np.sum(((noise - emcee_wave(theta, **wave_kwargs))/yerr) ** 2)

def lnprior(theta, ranges):
    if np.all(ranges[:,0] < theta) and np.all(theta < ranges[:,1]):
        return 0.0
    return -np.inf

def lnprob(theta, noise, yerr, ranges, wave_kwargs):
    lp = lnprior(theta, ranges)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, noise, yerr, wave_kwargs)
    
if __name__ == "__main__":
    # -------------------------------------
    REAL_AMPLITUDE = 7.0
    REAL_DAMPING = 0.3
    REAL_ANGULAR_FREQ = 1.2
    DATA_STEPS = 1000
    TIMESPAN = 10
    NOISE_AMPLITUDE = 0.0
    NOISE_SCALE = 0.0
    NDIM = 3
    NWALKERS = 50
    NUM_ITERATIONS = 100000
    RANGES = np.array(
        [
            [0.1, 10.0],    # amplitude (A) range
            [0.1, 3.0],     # damping (b) range
            [0.1, 10.0],    # angular frequency (omega) range
        ]
    )
    WAVE_KWARGS = {"phase": 0.0, "seconds": TIMESPAN, "steps": DATA_STEPS}
    # --------------------------------------
    
    time, real_wave = emcee_wave([REAL_AMPLITUDE, REAL_DAMPING, REAL_ANGULAR_FREQ], return_time=True, **WAVE_KWARGS)
    noise = generate_noise(real_wave, NOISE_SCALE, NOISE_AMPLITUDE)
    yerr = 0.05 * np.mean(noise)
    
    likelihood_kwargs = {
        "fcn": emcee_wave,
        "noise": noise,
        "fcn_kwargs": WAVE_KWARGS,
        "tol": 1e-9
    }
    lnprob_kwargs = {
        "noise": noise,
        "yerr": yerr,
        "ranges": RANGES,
        "wave_kwargs": WAVE_KWARGS
    }
    
    # Prior probabilities for the parameters for each walker
    priors = np.random.uniform(low=RANGES[:,0], high=RANGES[:,1], size=(NWALKERS, NDIM))

    metropolis_hastings_move = emcee.moves.MHMove(proposal_function=mh_proposal)
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, log_prob_fn=lnprob, kwargs=lnprob_kwargs)
    
    # Burn in
    state = sampler.run_mcmc(priors, 300)
    sampler.reset()
    # Actual run
    samples, prob, state = sampler.run_mcmc(state, NUM_ITERATIONS)

    # ------------- Plotting ------------------
    print("p0:", priors, "state:", state, "samples:", samples, samples.shape, sep="\n"*2)
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

    result_wave = emcee_wave([mean_amplitude, mean_damping, mean_angular_freq], **WAVE_KWARGS)
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
    axes["A"].set_xlim(left=RANGES[0, 0], right=RANGES[0, 1])

    axes["B"].hist(damping_samples, bins=100, density=True, label="estimated distribution")
    axes["B"].set_xlabel("Damping")
    ylim = axes["B"].get_ylim()
    axes["B"].plot([REAL_DAMPING, REAL_DAMPING], [0.0, ylim[1]], label = "true value")
    axes["B"].plot([mean_damping, mean_damping], [0.0, ylim[1]], label="sample mean")
    axes["B"].plot([mean_damping+std_damping, mean_damping+std_damping], [0.0, ylim[1]], "r--", label="+1 std")
    axes["B"].plot([mean_damping-std_damping, mean_damping-std_damping], [0.0, ylim[1]], "r--", label="-1 std")
    axes["B"].legend(prop={"size": 8})
    axes["B"].set_xlim(left=RANGES[1, 0], right=RANGES[1, 1])

    axes["F"].hist(angular_freq_samples, bins=100, density=True, label="estimated distribution")
    axes["F"].set_xlabel("Angular frequency")
    ylim = axes["F"].get_ylim()
    axes["F"].plot([REAL_ANGULAR_FREQ, REAL_ANGULAR_FREQ], [0.0, ylim[1]], label = "true value")
    axes["F"].plot([mean_angular_freq, mean_angular_freq], [0.0, ylim[1]], label="sample mean")
    axes["F"].plot([mean_angular_freq+std_angular_freq, mean_angular_freq+std_angular_freq], [0.0, ylim[1]], "r--", label="+1 std")
    axes["F"].plot([mean_angular_freq-std_angular_freq, mean_angular_freq-std_angular_freq], [0.0, ylim[1]], "r--", label="-1 std")
    axes["F"].legend(prop={"size": 8})
    axes["F"].set_xlim(left=RANGES[2, 0], right=RANGES[2, 1])
        
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


