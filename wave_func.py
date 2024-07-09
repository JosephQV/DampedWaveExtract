import numpy as np

# Returns only the displacement, for use by the MCMC class
def mcmc_wave(amplitude, damping=0.2, angular_freq=1.5, phase=0.0, seconds=30.0, steps=1000, return_time=False):
    time = np.linspace(0, seconds, steps)
    displacement = amplitude * np.exp(-1.0 * damping / 2.0 * time) * np.sin((angular_freq * time) + phase)
    if return_time:
        return time, displacement
    return displacement