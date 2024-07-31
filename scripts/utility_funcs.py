import numpy as np
import pathlib
from wave_funcs import *


def evaluate_wave_fcn(wave_fcn, theta: np.ndarray, wave_kwargs: dict, return_time: bool = False) -> np.ndarray:
    if type(wave_fcn) is str:
        fcn = wave_fcn
    else:
        fcn = wave_fcn.__name__
    
    if return_time == True:
        wave_kwargs.update({"return_time": return_time})
        time, wave = eval(fcn)(theta, **wave_kwargs)
        wave_kwargs.pop("return_time")
        return time, wave
    return eval(fcn)(theta, **wave_kwargs)


def generate_noise(data: np.ndarray, snr: float) -> np.ndarray:
    snr = max(snr, 0.01)
    ev_signal = np.mean(data ** 2)  # Expected value of signal variable
    ev_noise = ev_signal / snr      # Expected value of noise = EV(signal ** 2) / Signal to noise ratio (SNR)
    
    noise =  np.random.normal(loc=0.0, scale=ev_noise, size=len(data))
    return data + noise


def guess_params(param_ranges: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is not None:
        return rng.uniform(low=param_ranges[:,0], high=param_ranges[:,1], size=len(param_ranges))
    return np.random.uniform(low=param_ranges[:,0], high=param_ranges[:,1], size=len(param_ranges))


def compute_rms(observed: np.ndarray, predicted: np.ndarray) -> float:
    diff = observed - predicted
    return np.sqrt(np.mean(diff**2))
    
    
def print_sample_statuses(samples):
    for i, sample in enumerate(samples):
        current_params = sample
        acceptance_status = "Accepted"
        
        if i > 0:
            previous_params = samples[i - 1]
            
            if np.all(current_params == previous_params):
                acceptance_status = "Rejected (same as previous)"

        print(f"Trial {i+1}: ({acceptance_status})\n{current_params}")


def thin_samples(samples: np.ndarray, thin_percentage: float = 0.25, cutoff_percentage: float = 0.2) -> np.ndarray:
    """
    Thin the array of samples by skipping a percentage of them from the beginning and only keeping a given percentage of the remaining
    (chosen randomly).

    Args:
        samples (_type_): _description_
        thin_percentage (float, optional): percentage of values from samples to keep after thinning, 0.25 keeps 25% of the values. Defaults to 0.25.
        cutoff_percentage (float, optional): percentage of values from the beginning of samples to skip, 0.2 would start at 20% through the array. Defaults to 0.2.

    Returns:
        _type_: _description_
    """
    if thin_percentage == 0:
        return np.array([])
    start = int(samples.size * cutoff_percentage)
    count = int(thin_percentage * (samples.size - start))
    indices = np.random.randint(start, samples.size, count)
    return samples[indices]


def save_figure(figure, name: str):
    loc = pathlib.Path.cwd().parent.joinpath(f"figures/{name}")
    figure.savefig(loc)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plotting_funcs import FACECOLOR
    
    axes = plt.subplot(111)
    steps = 1000
    time = np.linspace(0, 60, steps)
    data = np.sin(time)
    
    SNRs = [1.00, 0.50, 0.25, 0.125]
    colors = ["#58fcc3", "#48cefa", "#faa823", "#fc654e"]
    
    splits = len(SNRs)
    for i in range(splits):
        part = int(steps / splits)
        noise = generate_noise(data[i * part : i * part + part], snr=SNRs[i])
        axes.plot(time[i * part : i * part + part], noise, color=colors[i % len(colors)], label=f"SNR: {SNRs[i]}")

    axes.plot(time, data, color="black", label="Ex. signal", alpha=0.6)
    axes.legend(loc="upper left", prop={"size": 12})
    axes.set_xlabel("time")
    axes.set_facecolor(FACECOLOR)
    axes.grid()
    plt.title("Some Signal to Noise Ratios (SNR)", fontdict={"family": "monospace", "size": "x-large"})
    plt.show()