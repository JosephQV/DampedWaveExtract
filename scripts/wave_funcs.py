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


def emcee_sine_gaussian_wave(theta, seconds=30, steps=1000, return_time=False):
    amplitude, omega, mean_time, std_dev = theta
    time = np.linspace(0, seconds, steps)
    displacement = amplitude * np.exp(-0.5 * (((time - mean_time) / std_dev) ** 2 )) * np.sin(omega * time)
    if return_time:
        return time, displacement
    return displacement


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    t, w1 = emcee_sine_gaussian_wave([3.0, 10.0, 10.0, 3.3], return_time=True)
    w2 = emcee_wave([7.0, 0.3, 1.5])
    
    plt.plot(t, w1, "r", t, w2, "g")
    plt.show()
    