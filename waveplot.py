import matplotlib.pyplot as plt
import numpy as np
import sys

def wave(amplitude, damping, angular_freq, freq=None, phase=0.0, seconds=30.0, steps=1000):
    time = np.linspace(0, seconds, steps)
    if freq:
        angular_freq = 2.0 * np.pi * freq
    exp = -1.0 * damping / 2.0
    displacement = amplitude * np.pow(np.e, exp * time) * np.sin((angular_freq * time) + phase)
    return time, displacement

out_file = sys.argv[1]

a = 7.0
b = 0.2
w = 2.0

t, d = wave(amplitude=a, damping=b, angular_freq=w)

plt.plot(t, d)
plt.xlabel("time (s)")
plt.ylabel("Amplitude (m)")
plt.annotate(f"A(0): {a}  omega: {w}  damping: {b}", (0.4, 0.9), xycoords="axes fraction") 
plt.grid()
plt.savefig(f"{out_file}")
