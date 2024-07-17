import numpy as np
import pathlib

def generate_noise(data: np.ndarray, scale: float, noise_amp: float = 1.0) -> np.ndarray:
    noise = 2 * noise_amp * np.random.normal(loc=0.0, scale=scale, size=len(data)) - noise_amp
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
    print(__file__)
    import matplotlib.pyplot as plt
    data = np.zeros(1000)
    noise = generate_noise(data, 1.5, 5.0)
    plt.plot(np.linspace(0, 30, 1000), noise)
    plt.show()