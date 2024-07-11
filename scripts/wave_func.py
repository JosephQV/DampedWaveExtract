import numpy as np

# Default returns only the displacement, for use by the MCMC class
def mcmc_wave(amplitude, damping, angular_freq, phase=0.0, seconds=30, steps=1000, return_time=False):
    time = np.linspace(0, seconds, steps)
    displacement = amplitude * np.exp(-1.0 * damping / 2.0 * time) * np.sin((angular_freq * time) + phase)
    if return_time:
        return time, displacement
    return displacement

# Different header to satisfy 
def emcee_wave(*args, **kwargs):
    amplitude = args[0]
    damping = args[1]
    angular_freq = args[2]
    phase = kwargs["phase"]
    steps = kwargs["steps"]
    seconds = kwargs["seconds"]
    return_time = kwargs["return_time"]
    
    time = np.linspace(0, seconds, steps)
    displacement = amplitude * np.exp(-1.0 * damping / 2.0 * time) * np.sin((angular_freq * time) + phase)
    if return_time:
        return time, displacement
    return displacement