import numpy as np
import matplotlib.pyplot as plt
from utility_funcs import evaluate_wave_fcn, generate_noise, compute_rms


FACECOLOR = "#e8e1da"

class PlottingWrapper:
    def __init__(
        self,
        samples: np.ndarray,
        real_theta: np.ndarray,
        best_theta: np.ndarray,
        ranges: np.ndarray,
        names,
        labels,
        wave_fcn,
        wave_kwargs: dict,
        snr: float
    ):
        self.samples = samples
        
        self.real_theta = real_theta
        self.best_theta = best_theta
        
        self.ranges = ranges
        self.names = names
        self.labels = labels

        self.wave_fcn = wave_fcn
        self.wave_kwargs = wave_kwargs
        self.snr = snr
        
        self.time, self.real_wave = evaluate_wave_fcn(self.wave_fcn, self.real_theta, self.wave_kwargs, return_time=True)
        self.noise = generate_noise(self.real_wave, self.snr)
        self.means = np.mean(self.samples, axis=0)
        self.medians = np.median(self.samples, axis=0)
        self.stds = np.std(self.samples, axis=0)
        self.rms = compute_rms(
            observed=self.real_wave,
            predicted=evaluate_wave_fcn(self.wave_fcn, self.best_theta, self.wave_kwargs)
        )
        
        self.axes_facecolor = FACECOLOR
    
    def _plot_wave(self, axes, theta: np.ndarray, plot_kwargs: dict | None = {}):
        wave = evaluate_wave_fcn(self.wave_fcn, theta, self.wave_kwargs)
        axes.plot(self.time, wave, **plot_kwargs)
        
    def _make_figure(self, title, size=(8, 5), fontsize=18):
        figure = plt.figure(figsize=size, dpi=200)
        figure.suptitle(title, fontsize=fontsize, fontfamily="monospace")
        return figure
        
    def _plot_distribution(self, axes, parameter: int):
        trace = self.samples[:, parameter]
        real = self.real_theta[parameter]
        
        mean = self.means[parameter]
        median = self.medians[parameter]
        std = self.stds[parameter]
        
        axes.hist(trace, bins=100, density=True, label="estimated distribution")
        axes.set_xlabel(self.names[parameter])
        ybound = axes.get_ybound()
        axes.plot([real, real], [0.0, ybound[1]], label="true value", color="black")
        axes.plot([median, median], [0.0, ybound[1]], label="sample median", color="green")
        axes.plot([mean+std, mean+std], [0.0, ybound[1]], "r--", label="+-1 std")
        axes.plot([mean-std, mean-std], [0.0, ybound[1]], "r--")
        axes.legend(loc="upper right", prop={"size": 8}, framealpha=0.5)
        axes.set_xlim(left=self.ranges[parameter, 0], right=self.ranges[parameter, 1])

    def _fill_within_std(self, axes):
        above = evaluate_wave_fcn(self.wave_fcn, self.means+self.stds, self.wave_kwargs)
        below = evaluate_wave_fcn(self.wave_fcn, self.means-self.stds, self.wave_kwargs)
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
            self._plot_wave(axes, thetas_within_std[i], plot_kwargs={"color": "green", "linestyle": "-", "alpha": 0.05})

    def _plot_posterior_trace(self, axes, parameter, trace, iter_start, iter_end):
        x = np.arange(iter_start, iter_end, 1)
        trace = trace[iter_start:iter_end]
        axes.plot(x, trace, color="red", label="chain", alpha=0.5)
        axes.set_ylabel(self.names[parameter], labelpad=2.0)
        axes.set_ylim(self.ranges[parameter])
        axes.plot([iter_start, iter_end], [self.real_theta[parameter], self.real_theta[parameter]], color="green", alpha=0.7, label="real")
        axes.legend(loc="upper right", prop={"size": 8}, framealpha=0.5)
        axes.grid()
        axes.set_facecolor(self.axes_facecolor)


    def plot_mcmc_wave_results(self):
        """
        Create a figure with the real and generated data, the noise, and the sample distributions for 3 parameters.
        """
        figure = self._make_figure(title="Extracting Signal Parameters From Noise", size=(16, 10))
        axes = figure.subplot_mosaic(
        """
        WWN
        ABF
        """,
        subplot_kw={"facecolor": self.axes_facecolor}
        )

        self._plot_distribution(axes["A"], 0)
        self._plot_distribution(axes["B"], 1)
        self._plot_distribution(axes["F"], 2)
        
        self._plot_wave(axes["W"], self.real_theta, plot_kwargs={"color": "red", "label": "real"})
        self._plot_wave(axes["W"], self.medians, plot_kwargs={"color": "green", "label": "sample median"})
        self._plot_wave(axes["W"], self.best_theta, plot_kwargs={"color": "blue", "label": "highest likelihood"})        
        
        axes["W"].annotate(f"Amplitude: {self.real_theta[0]}\nDamping: {self.real_theta[1]}\nAngular Frequency: {self.real_theta[2]}\n\nError (RMS): {self.rms:.3f}", xy=(0.75, 0.50), xycoords="axes fraction")
        axes["W"].legend()
        axes["W"].set_xlabel("time")
        axes["W"].grid()
        axes["N"].plot(self.time, self.noise, label="noise")
        axes["N"].set_xlabel("time")
        axes["N"].legend()
        axes["N"].annotate(f"$SNR = ${self.snr}", xy=(0.7, 0.07), xycoords="axes fraction")
        axes["N"].grid()
        
        return figure, axes


    def plot_sample_distributions(self, title: str = "Probability Distributions for Each Parameter"):
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
            self._plot_distribution(axes[indices[p]], parameter=p, xlabel=self.names[p])
        
        figure.subplots_adjust(top=0.92)
        return figure, axes

    def plot_real_vs_generated(self, title: str = "Real and Estimated Signal"):
        """
        Create a figure showing the real wave signal and the generated signals based on estimations from the samples within 1 standard deviation.
        """
        figure = self._make_figure(title)
        
        axes = figure.add_subplot()
        axes.set_xlabel("time")
        axes.set_facecolor(self.axes_facecolor)
        axes.set_ylim(-1*self.real_theta[0]*2, self.real_theta[0]*2)
                
        annotation = ""
        for p in range(len(self.labels)):
            annotation += f"{self.labels[p]} = {self.real_theta[p]}\n"
            
        annotation += f"\nError (RMS): {self.rms:.3f}"
        axes.annotate(annotation, xy=(0.75, 0.70), xycoords="axes fraction")
        
        self._plot_wave(axes, self.real_theta, plot_kwargs={"color": "red", "label": "real"})
        self._plot_wave(axes, self.medians, plot_kwargs={"color": "green", "label": "sample median", "alpha": 0.4})
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
        self._plot_wave(axes, self.real_theta, plot_kwargs={"color": "red", "label": "real", "alpha": 0.5})
        axes.legend(loc="upper left", prop={"size": 10}, framealpha=0.75)
        axes.grid() 
        axes.annotate(f"$SNR = ${self.snr}", xy=(0.1, 0.1), xycoords="axes fraction")
        
        return figure, axes
    
    def plot_posterior_traces(self, single_chain, iter_start=0, iter_end=None, title: str = "Convergence of Sample Chains"):
        figure = self._make_figure(title, size=(8, 8), fontsize=14)
        
        ndim = self.samples.shape[1]
        axes = figure.subplots(ndim, 1)
        
        if iter_end is None:
            iter_end = single_chain.shape[0]
        for p in range(ndim):
            self._plot_posterior_trace(axes[p], parameter=p, trace=single_chain[:, p], iter_start=iter_start, iter_end=iter_end)
        
        figure.subplots_adjust(bottom=0.08, top=0.92, hspace=0.4)
        return figure, axes





    

