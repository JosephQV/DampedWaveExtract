import numpy as np
import matplotlib.pyplot as plt
from wave_funcs import emcee_wave
from utility_funcs import compute_rms


def plot_distribution(axes, samples: np.ndarray, real: np.ndarray, xlabel: str, xlim):
    mean = np.mean(samples)
    std = np.std(samples)
    axes.hist(samples, bins=100, density=True, label="estimated distribution")
    axes.set_xlabel(xlabel)
    ylim = axes.get_ylim()
    axes.plot([real, real], [0.0, ylim[1]], label = "true value")
    axes.plot([mean, mean], [0.0, ylim[1]], label="sample mean")
    axes.plot([mean+std, mean+std], [0.0, ylim[1]], "r--", label="+-1 std")
    axes.plot([mean-std, mean-std], [0.0, ylim[1]], "r--")
    axes.legend(prop={"size": 8})
    axes.set_xlim(left=xlim[0], right=xlim[1])


def plot_waves_within_std(axes, samples: np.ndarray, x: np.ndarray, wave_kwargs: dict, num_lines: int = 300):
    thetas_within_std = np.zeros_like(samples)
    min_length = 10e10
    stds = np.zeros(shape=samples.shape[1])
    means = np.zeros(shape=samples.shape[1])
    for param in range(samples.shape[1]):
        param_samples = samples[:, param]
        mean = np.mean(param_samples)
        means[param] = mean
        std = np.std(param_samples)
        stds[param] = std
        
        param_samples = param_samples[np.where(param_samples > (mean - std))]
        param_samples = param_samples[np.where(param_samples < (mean + std))]

        thetas_within_std[0:len(param_samples), param] = param_samples
        min_length = min(min_length, len(param_samples))
    
    axes.plot(x, emcee_wave(theta=means+stds, **wave_kwargs), "g-", alpha=0.10)
    axes.plot(x, emcee_wave(theta=means-stds, **wave_kwargs), "g-", alpha=0.10)
    
    min_length = min(min_length, num_lines)
    for theta in range(min_length):
        axes.plot(x, emcee_wave(theta=thetas_within_std[theta], **wave_kwargs), "g-", alpha=0.01)


def plot_mcmc_wave_results(samples: np.ndarray, real_data: np.ndarray, real_params: np.ndarray, param_ranges: np.ndarray, wave_kwargs: dict, noise: np.ndarray, x: np.ndarray):
    amplitude_samples = samples[:, 0]
    damping_samples = samples[:, 1]
    angular_freq_samples = samples[:, 2]

    mean_amplitude = np.mean(amplitude_samples)
    mean_damping = np.mean(damping_samples)
    mean_angular_freq = np.mean(angular_freq_samples)

    figure = plt.figure(figsize=(13, 7), layout="constrained")
    figure.suptitle("(emcee) Extracting Signal Parameters From Noise")
    axes = figure.subplot_mosaic(
    """
    WWN.
    ABF.
    """,
    width_ratios=[1, 1, 1, 0.05]
    )

    plot_distribution(axes["A"], amplitude_samples, real_params[0], "Amplitude", param_ranges[0])
    plot_distribution(axes["B"], damping_samples, real_params[1], "Damping", param_ranges[1])
    plot_distribution(axes["F"], angular_freq_samples, real_params[2], "Angular Frequency", param_ranges[2])
    
    axes["W"].plot(x, real_data, "r", label="real")
    axes["W"].set_xlabel("time")
    result_wave = emcee_wave([mean_amplitude, mean_damping, mean_angular_freq], **wave_kwargs)
    rms_deviation = compute_rms(real_data, result_wave)
    axes["W"].annotate(f"Real Parameters:\nAmplitude: {real_params[0]}\nDamping: {real_params[1]}\nAngular Frequency: {real_params[2]}\n\nRMS Deviation: {rms_deviation:.3f}", xy=(0.75, 0.6), xycoords="axes fraction")
    axes["W"].plot(x, result_wave, "g", label="generated")
    plot_waves_within_std(axes["W"], samples, x, wave_kwargs)
    axes["W"].legend()

    axes["N"].plot(x, noise, label="noise")
    axes["N"].set_xlabel("time")
    axes["N"].legend()
    # axes["N"].annotate(f"Noise amplitude: {np.max(noise[int(len(noise) * 0.75):]) - np.mean(noise[int(noise.shape[0] * 0.75)])}", xy=(0.53, 0.9), xycoords="axes fraction")
    
    return figure, axes