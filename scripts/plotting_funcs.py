import numpy as np
import matplotlib.pyplot as plt
from wave_funcs import *
from utility_funcs import compute_rms


def plot_mcmc_wave_results(samples: np.ndarray, real_data: np.ndarray, real_params: np.ndarray, param_ranges: np.ndarray, wave_kwargs: dict, wave_fcn, noise: np.ndarray, x: np.ndarray):
    """
    Create a figure with the real and generated data, the noise, and the sample distributions for 3 parameters.
    """
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

    _plot_distribution(axes["A"], amplitude_samples, real_params[0], "Amplitude", param_ranges[0])
    _plot_distribution(axes["B"], damping_samples, real_params[1], "Damping", param_ranges[1])
    _plot_distribution(axes["F"], angular_freq_samples, real_params[2], "Angular Frequency", param_ranges[2])
    
    axes["W"].plot(x, real_data, "r", label="real")
    axes["W"].set_xlabel("time")
    result_wave = emcee_wave([mean_amplitude, mean_damping, mean_angular_freq], **wave_kwargs)
    rms_deviation = compute_rms(real_data, result_wave)
    axes["W"].annotate(f"Real Parameters:\nAmplitude: {real_params[0]}\nDamping: {real_params[1]}\nAngular Frequency: {real_params[2]}\n\nRMS Deviation: {rms_deviation:.3f}", xy=(0.75, 0.6), xycoords="axes fraction")
    axes["W"].plot(x, result_wave, "g", label="generated")
    _fill_within_std(axes["W"], samples, x, wave_kwargs, wave_fcn)
    axes["W"].legend()

    axes["N"].plot(x, noise, label="noise")
    axes["N"].set_xlabel("time")
    axes["N"].legend()
    # axes["N"].annotate(f"Noise amplitude: {np.max(noise[int(len(noise) * 0.75):]) - np.mean(noise[int(noise.shape[0] * 0.75)])}", xy=(0.53, 0.9), xycoords="axes fraction")
    
    return figure, axes


def plot_sample_distributions(samples: np.ndarray, real_theta: np.ndarray, xlabels: list[str], ranges: np.ndarray, title: str = "Probability Distributions for Each Parameter"):
    """
    Create a figure with a plotted posterior distribution for each parameter in samples.
    """
    figure = _make_figure(title, size=(8, 8))
    
    ndim = samples.shape[1]
    indices = {
        0: (0,0),
        1: (0,1),
        2: (1,0),
        3: (1,1)
    }
    axes = figure.subplots(2, 2)
    
    for p in range(ndim):
        param = samples[:, p]
        _plot_distribution(axes[indices[p]], samples=param, real=real_theta[p], xlabel=xlabels[p], xlim=ranges[p])
    
    return figure, axes


def plot_real_vs_generated(samples: np.ndarray, real_theta: np.ndarray, x: np.ndarray, xlabel: str, wave_kwargs: dict, wave_fcn, title: str = "Real and Estimated Signal", annotation: str = ""):
    """
    Create a figure showing the real wave signal and the generated signals based on estimations from the samples within 1 standard deviation.
    """
    figure = _make_figure(title)
    
    axes = figure.add_subplot()
    axes.set_xlabel(xlabel)
    
    mean_theta = np.mean(samples, axis=0)
    real_data = eval(wave_fcn.__name__)(real_theta, **wave_kwargs)
    result_data = eval(wave_fcn.__name__)(mean_theta, **wave_kwargs)
    rms_deviation = compute_rms(real_data, result_data)
    
    text = f"{annotation}\nError (RMS): {rms_deviation:.3f}"
    axes.annotate(text, xy=(0.75, 0.6), xycoords="axes fraction")
    
    _plot_wave(axes, x, wave_fcn, real_theta, wave_kwargs, fmt="r", label="real")    
    _fill_within_std(axes, samples, x, wave_kwargs, wave_fcn=wave_fcn)
    _plot_waves_within_std(axes, samples, x, wave_kwargs, wave_fcn=wave_fcn)
    axes.legend()

    return figure, axes


def plot_signal_in_noise(noise: np.ndarray, real_theta: np.ndarray, x: np.ndarray, xlabel: str, wave_kwargs: dict, wave_fcn, title: str = "Signal Masked in Noise"):
    """
    Create a figure showing the real wave signal within the random noise.
    """
    figure = _make_figure(title)
    
    axes = figure.add_subplot()
    axes.set_xlabel(xlabel)
    
    axes.plot(x, noise, label="noise")
    _plot_wave(axes, x, wave_fcn, real_theta, wave_kwargs, fmt="r", label="real")
    axes.legend()   
    
    return figure, axes


def _make_figure(title, size=(8, 5)):
    figure = plt.figure(figsize=size)
    figure.suptitle(title, fontsize="x-large", fontfamily="monospace")
    return figure


def _plot_wave(axes, x: np.ndarray, wave_fcn, theta: np.ndarray, wave_kwargs: dict, fmt: str = "r", label: str = None):
    data = eval(wave_fcn.__name__)(theta, **wave_kwargs)
    if label:
        axes.plot(x, data, fmt, label=label)
    else:
        axes.plot(x, data, fmt)
    

def _plot_distribution(axes, samples: np.ndarray, real: np.ndarray, xlabel: str, xlim: np.ndarray):
    mean = np.mean(samples)
    std = np.std(samples)
    
    axes.hist(samples, bins=100, density=True, label="estimated distribution")
    axes.set_xlabel(xlabel)
    ylim = axes.get_ylim()
    axes.plot([real, real], [0.0, ylim[1]], label="true value", color="black")
    axes.plot([mean, mean], [0.0, ylim[1]], label="sample mean", color="red")
    axes.plot([mean+std, mean+std], [0.0, ylim[1]], "r--", label="+-1 std")
    axes.plot([mean-std, mean-std], [0.0, ylim[1]], "r--")
    axes.legend(prop={"size": 8})
    axes.set_xlim(left=xlim[0], right=xlim[1])


def _fill_within_std(axes, samples: np.ndarray, x: np.ndarray, wave_kwargs: dict, wave_fcn):
    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)
    
    above = eval(wave_fcn.__name__)(means+stds, **wave_kwargs)
    below = eval(wave_fcn.__name__)(means-stds, **wave_kwargs)
    
    axes.fill_between(x, above, below, alpha=0.4, label="within 1 std")
    

def _plot_waves_within_std(axes, samples: np.ndarray, x: np.ndarray, wave_kwargs: dict, wave_fcn, num_lines: int = 300):
    thetas_within_std = np.zeros_like(samples)
    min_length = 10e10
    
    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)
    
    for p in range(samples.shape[1]):
        param_samples = samples[:, p]
        
        param_samples = param_samples[np.where(param_samples > (means[p] - stds[p]))]
        param_samples = param_samples[np.where(param_samples < (means[p] + stds[p]))]
        thetas_within_std[0:len(param_samples), p] = param_samples
        min_length = min(min_length, len(param_samples))
    
    theta = means + stds
    axes.plot(x, eval(wave_fcn.__name__)(theta, **wave_kwargs), "g-", alpha=0.10)
    theta = means - stds
    axes.plot(x, eval(wave_fcn.__name__)(theta, **wave_kwargs), "g-", alpha=0.10)
    
    min_length = min(min_length, num_lines)
    for theta in range(min_length):
        axes.plot(x, eval(wave_fcn.__name__)(theta, **wave_kwargs), "g-", alpha=0.01)
