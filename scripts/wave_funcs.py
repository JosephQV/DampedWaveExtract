import numpy as np


# For use by the custom MCMC class
def mcmc_wave(amplitude, damping, angular_freq, phase=0.0, seconds=30, steps=1000, return_time=False):
    time = np.linspace(0, seconds, steps)
    displacement = amplitude * np.exp(-1.0 * damping / 2.0 * time) * np.sin((angular_freq * time) + phase)
    if return_time:
        return time, displacement
    return displacement


# For use when working with the emcee package
def emcee_wave(theta, phase=0.0, seconds=30, steps=1000, return_time=False):
    amplitude, damping, angular_freq = theta
    time = np.linspace(0, seconds, steps)
    displacement = amplitude * np.exp(-1.0 * damping / 2.0 * time) * np.sin((angular_freq * time) + phase)
    if return_time:
        return time, displacement
    return displacement


if __name__ == "__main__":
    w = mcmc_wave(5, 800, 2.0)
    print(w, w.size)