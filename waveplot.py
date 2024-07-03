import matplotlib.pyplot as plt
import numpy as np
import math

def wave(A, w, b, o=0, seconds=30):
    steps = 1000
    time = np.linspace(0, seconds, steps)
    displacement = np.zeros_like(time)
    for t in range(steps):
        displacement[t] = A * math.pow(math.e, (-1 * b * time[t] / 2)) * math.cos(w * time[t] + o)
    return time, displacement

A = 7
w = 0.8
b = 0.25


t, D = wave(A, w, b)

plt.plot(t, D)
plt.xlabel("time (s)")
plt.ylabel("Amplitude (m)")
plt.annotate(f"A(0): {A} w: {w} b: {b}", (0.4, 0.9), xycoords="axes fraction") 
plt.grid()
plt.savefig("wave.png")
