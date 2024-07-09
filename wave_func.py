import numpy as np

def wave(amplitude, damping, angular_freq, freq=None, phase=0.0, seconds=30.0, steps=1000):
    time = np.linspace(0, seconds, steps)
    if freq:
        angular_freq = 2.0 * np.pi * freq
    
    displacement = amplitude * np.exp(-1.0 * damping / 2.0 * time) * np.sin((angular_freq * time) + phase)
    return time, displacement

# Returns only the displacement, for use by the MCMC class
def mcmc_wave(amplitude, damping=0.2, angular_freq=1.5, phase=0.0, seconds=30.0, steps=1000):
    time = np.linspace(0, seconds, steps)
    displacement = amplitude * np.exp(-1.0 * damping / 2.0 * time) * np.sin((angular_freq * time) + phase)
    return displacement