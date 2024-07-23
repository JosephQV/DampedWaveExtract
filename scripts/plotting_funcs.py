import numpy as np
import matplotlib.pyplot as plt
from utility_funcs import compute_rms, evaluate_wave_fcn
from emcee_funcs import compare_for_error


FACECOLOR = "#e8e1da"

class PlottingWrapper:
    def __init__(
        self,
        samples: np.ndarray,
        param_ranges: np.ndarray,
        wave_fcn,
        wave_kwargs: dict,
        real_parameters: np.ndarray,
        noise: np.ndarray,
        snr: float
    ):
        self.samples = samples
        self.param_ranges = param_ranges
        self.wave_fcn = wave_fcn
        self.wave_kwargs = wave_kwargs
        self.real_parameters = real_parameters
        self.noise = noise
        self.snr = snr
        self.time, self.real_wave = self._get_wave(self.real_parameters, return_time=True)
        self.means = np.mean(self.samples, axis=0)
        self.medians = np.median(self.samples, axis=0)
        self.stds = np.std(self.samples, axis=0)
        self.mean_generated_wave = self._get_wave(self.means)
        self.median_generated_wave = self._get_wave(self.medians)
        self.axes_facecolor = FACECOLOR
    
    def _get_wave(self, theta, return_time: bool = False):
        if return_time == True:
            self.wave_kwargs.update({"return_time": return_time})
            time, wave = evaluate_wave_fcn(self.wave_fcn, theta, self.wave_kwargs)
            self.wave_kwargs.pop("return_time")
            return time, wave
        return evaluate_wave_fcn(self.wave_fcn, theta, self.wave_kwargs)
    
    def _plot_wave(self, axes, theta: np.ndarray, plot_kwargs: dict | None = {}):
        wave = self._get_wave(theta)
        axes.plot(self.time, wave, **plot_kwargs)
        
    def _make_figure(self, title, size=(8, 5)):
        figure = plt.figure(figsize=size)
        figure.suptitle(title, fontsize="x-large", fontfamily="monospace")
        return figure
        
    def _plot_distribution(self, axes, parameter: int, xlabel: str):
        parameter_samples = self.samples[parameter]
        real_parameter = self.real_parameters[parameter]
        
        mean = self.means[parameter]
        median = self.medians[parameter]
        std = self.stds[parameter]
        
        axes.hist(parameter_samples, bins=50, density=True, label="estimated distribution")
        axes.set_xlabel(xlabel)
        ylim = axes.get_ylim()
        axes.plot([real_parameter, real_parameter], [0.0, ylim[1]], label="true value", color="black")
        axes.plot([median, median], [0.0, ylim[1]], label="sample median", color="green")
        axes.plot([mean+std, mean+std], [0.0, ylim[1]], "r--", label="+-1 std")
        axes.plot([mean-std, mean-std], [0.0, ylim[1]], "r--")
        axes.legend(loc="upper right", prop={"size": 8}, framealpha=0.5)
        axes.set_xlim(left=self.param_ranges[parameter, 0], right=self.param_ranges[parameter, 1])

    def _fill_within_std(self, axes):
        
        above = self._get_wave(self.means+self.stds)
        below = self._get_wave(self.means-self.stds)
        
        axes.fill_between(self.time, above, below, alpha=0.6, label="within 1 std")
        
    def _plot_waves_within_std(self, axes, num_lines: int = 300):
        thetas_within_std = np.zeros_like(self.samples)
        min_length = 10e10
        
        for p in range(self.samples.shape[1]):
            param_samples = self.samples[:, p]
            
            param_samples = param_samples[np.where(param_samples > (self.means[p] - self.stds[p]))]
            param_samples = param_samples[np.where(param_samples < (self.means[p] + self.stds[p]))]
            thetas_within_std[0:len(param_samples), p] = param_samples
            min_length = min(min_length, len(param_samples))
        
        self._plot_wave(axes, self.means+self.stds, plot_kwargs={"color": "green", "linestyle": "-", "alpha": 0.1})
        self._plot_wave(axes, self.means-self.stds, plot_kwargs={"color": "green", "linestyle": "-", "alpha": 0.1})
        
        min_length = min(min_length, num_lines)
        for i in range(min_length):
            self._plot_wave(axes, thetas_within_std[i], plot_kwargs={"color": "green", "linestyle": "-", "alpha": 0.01})

    def plot_mcmc_wave_results(self):
        """
        Create a figure with the real and generated data, the noise, and the sample distributions for 3 parameters.
        """
        figure = plt.figure(figsize=(13, 7), layout="constrained")
        figure.suptitle("Extracting Signal Parameters From Noise")
        axes = figure.subplot_mosaic(
        """
        WWN.
        ABF.
        """,
        width_ratios=[1, 1, 1, 0.05]
        )

        self._plot_distribution(axes["A"], 0, "Amplitude")
        self._plot_distribution(axes["B"], 1, "Damping")
        self._plot_distribution(axes["F"], 2, "Angular Frequency")
        
        self._plot_wave(axes["W"], self.real_parameters, plot_kwargs={"color": "red", "label": "real"})
        self._plot_wave(axes["W"], self.medians, plot_kwargs={"color": "green", "label": "generated"})        
        
        rms_deviation = compare_for_error(self.real_parameters, self.samples, self.wave_fcn, self.wave_kwargs)
        axes["W"].annotate(f"Real Parameters:\nAmplitude: {self.real_parameters[0]}\nDamping: {self.real_parameters[1]}\nAngular Frequency: {self.real_parameters[2]}\n\nRMS Deviation: {rms_deviation:.3f}", xy=(0.75, 0.60), xycoords="axes fraction")
        #self._fill_within_std(axes["W"])
        self._plot_waves_within_std(axes["W"])
        axes["W"].legend()
        axes["W"].set_xlabel("time")
        axes["N"].plot(self.time, self.noise, label="noise")
        axes["N"].set_xlabel("time")
        axes["N"].legend()
        axes["N"].annotate(f"$SNR = ${self.snr}", xy=(0.53, 0.9), xycoords="axes fraction")
        
        return figure, axes


    def plot_sample_distributions(self, xlabels: list[str], title: str = "Probability Distributions for Each Parameter"):
        """
        Create a figure with a plotted posterior distribution for each parameter in samples.
        """
        figure = self._make_figure(title, size=(8, 8))
        
        ndim = self.samples.shape[1]
        indices = {
            0: (0,0),
            1: (0,1),
            2: (1,0),
            3: (1,1)
        }
        axes = figure.subplots(2, 2, subplot_kw={"facecolor": self.axes_facecolor})
        
        for p in range(ndim):
            self._plot_distribution(axes[indices[p]], parameter=p, xlabel=xlabels[p])
        
        return figure, axes

    def plot_real_vs_generated(self, title: str = "Real and Estimated Signal", annotation: str = ""):
        """
        Create a figure showing the real wave signal and the generated signals based on estimations from the samples within 1 standard deviation.
        """
        figure = self._make_figure(title)
        
        axes = figure.add_subplot()
        axes.set_xlabel("time")
        axes.set_facecolor(self.axes_facecolor)
        axes.set_ylim(-1*self.real_parameters[0]*2, self.real_parameters[0]*2)
        
        rms_deviation = compare_for_error(real_theta=self.real_parameters, samples=self.samples, wave_fcn=self.wave_fcn, wave_kwargs=self.wave_kwargs)
        
        text = f"{annotation}\nError (RMS): {rms_deviation:.3f}"
        axes.annotate(text, xy=(0.75, 0.70), xycoords="axes fraction")
        
        self._plot_wave(axes, self.real_parameters, plot_kwargs={"color": "red", "label": "real"})
        self._plot_wave(axes, self.medians, plot_kwargs={"color": "green", "label": "sample median", "alpha": 0.4})
        #self._fill_within_std(axes)
        #self._plot_waves_within_std(axes)
        axes.legend(loc="upper left", prop={"size": 10}, framealpha=0.5)
        axes.grid()

        return figure, axes

    def plot_signal_in_noise(self, title: str = "Signal Masked in Noise"):
        """
        Create a figure showing the real wave signal within the random noise.
        """
        figure = self._make_figure(title)
        
        axes = figure.add_subplot()
        axes.set_xlabel("time")
        axes.set_facecolor(self.axes_facecolor)
        
        axes.plot(self.time, self.noise, label="noise")
        self._plot_wave(axes, self.real_parameters, plot_kwargs={"color": "red", "label": "real", "alpha": 0.5})
        axes.legend(loc="upper left", prop={"size": 10}, framealpha=0.75)
        axes.grid() 
        axes.annotate(f"$SNR = ${self.snr}", xy=(0.1, 0.1), xycoords="axes fraction")
        
        return figure, axes






    

