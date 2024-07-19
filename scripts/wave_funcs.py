import numpy as np


def damped_wave(theta, phase=0.0, seconds=30, steps=1000, return_time=False):
    amplitude, damping, angular_freq = theta
    time = np.linspace(0, seconds, steps)
    displacement = amplitude * np.exp(-1.0 * damping / 2.0 * time) * np.sin((angular_freq * time) + phase)
    if return_time:
        return time, displacement
    return displacement


def sine_gaussian_wave(theta, seconds=30, steps=1000, return_time=False):
    amplitude, omega, mean_time, std_dev = theta
    time = np.linspace(0, seconds, steps)
    displacement = amplitude * np.exp(-0.5 * (((time - mean_time) / std_dev) ** 2 )) * np.sin(omega * time)
    if return_time:
        return time, displacement
    return displacement


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plotting_funcs import FACECOLOR
    
    damped_wave_theta = [7.0, 0.3, 1.5]
    sine_gaussian_wave_theta = [3.0, 7.0, 15.0, 3.5]
    kwargs = {"seconds": 30, "steps": 1000}
    
    ds = damped_wave(damped_wave_theta, **kwargs)
    time, sg = sine_gaussian_wave(sine_gaussian_wave_theta, return_time=True, **kwargs)
    
    axes = plt.subplot(211)
    axes.plot(time, ds, color="blue")
    axes.set_title("Damped Sinusoidal", fontdict={"family": "monospace"})
    axes.set_facecolor(FACECOLOR)
    axes.grid()
    ds_annotation = f"$A = ${damped_wave_theta[0]}\n$\\beta = ${damped_wave_theta[1]}\n$\\omega = ${damped_wave_theta[2]}"
    axes.annotate(ds_annotation, xy=(0.85, 0.7), xycoords="axes fraction", size=12)
    
    axes = plt.subplot(212)
    axes.plot(time, sg, color="red")
    axes.set_title("Sine-Gaussian", fontdict={"family": "monospace"})
    axes.set_xlabel("time")
    axes.set_facecolor(FACECOLOR)
    axes.grid()
    sg_annotation = f"$A = ${sine_gaussian_wave_theta[0]}\n$\\omega = ${sine_gaussian_wave_theta[1]}\n$\\mu = ${sine_gaussian_wave_theta[2]}\n$\\sigma = ${sine_gaussian_wave_theta[3]}"
    axes.annotate(sg_annotation, xy=(0.85, 0.65), xycoords="axes fraction", size=12)
    
    plt.suptitle("Signal Models", family="monospace", size="xx-large")
    plt.show()
    