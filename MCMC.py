import numpy as np
import functools

class MCMCModel:
    def __init__(self, function, param_ranges: np.ndarray, function_kwargs: dict = None):
        self.function = function
        self.param_ranges = param_ranges
        self.function_kwargs = function_kwargs
    
    def generate_noise(self, data, noise_amp=1.0):
        """
        Generates noisy data by adding random noise.

        Args:
        - x: numpy array of true values
        - A: amplitude of noise

        Returns:
        - numpy array of noisy data
        """
        noise = 2 * noise_amp * np.random.random(size=len(data)) - noise_amp
        return data + noise

    def compute_rms(self, observed, predicted):
        diff = observed - predicted
        return np.sqrt(np.mean(diff**2))

    def guess_params(self):
        return np.random.uniform(low=self.param_ranges[:,0], high=self.param_ranges[:,1], size=len(self.param_ranges))
        
    def likelihood(self, noisy_vals, params, tol=1e-9):
        trial_vals = eval(self.function.__name__)(*params, **self.function_kwargs)
        rms_error = self.compute_rms(noisy_vals, trial_vals)
        likelihood = 1 / (rms_error + tol) 
        return likelihood

    def metropolis_hastings(self, data, num_iterations, noise_amplitude=1.0):
        noisy_vals = self.generate_noise(data, noise_amplitude)

        current_params = self.guess_params()

        current_likelihood = self.likelihood(noisy_vals, current_params)

        samples = np.zeros(shape=(num_iterations, len(self.param_ranges)))
        samples[0] = current_params
        
        for i in range(1, num_iterations):
            proposed_params = self.guess_params()
            proposed_likelihood = self.likelihood(noisy_vals, proposed_params)

            acceptance_ratio = proposed_likelihood / current_likelihood
            if acceptance_ratio > np.random.uniform(0.7, 1.0):
                current_params = proposed_params
                current_likelihood = proposed_likelihood

            samples[i] = current_params

        return samples
    
    def thin_samples(self, samples, thin_ratio: float = 0.2, cutoff_start: int = 0):
        cutoff = samples[cutoff_start:]
        thinned = np.zeros_like(cutoff)
        i = 0
        mod = int(1/thin_ratio)
        for j in range(cutoff.size):
            if j % mod == 0:
                thinned[i] = cutoff[j]
                i += 1
        return np.trim_zeros(thinned)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from wave_func import mcmc_wave
    REAL_AMPLITUDE = 6.0
    REAL_DAMPING = 0.3
    REAL_OMEGA = 2.0
    REAL_PHASE = 0.0
    NOISE_AMPLITUDE = 4.0
    NUM_ITERATIONS = 100000
    DATA_STEPS = 10000
    THIN_RATIO = 0.1
    
    time, real_wave = mcmc_wave(REAL_AMPLITUDE, REAL_DAMPING, REAL_OMEGA, steps=DATA_STEPS, return_time=True)
    
    ranges = np.array(
        [
            [0.1, 10.0],    # amplitude (A) range
            [0.1, 3.0],     # damping (b) range
            [0.1, 10.0],    # angular frequency (omega) range
            # [0.0, 2*np.pi]  # phase constant range
        ]
    )
    kwargs = {"seconds": 30.0, "steps": DATA_STEPS, "return_time": False}

    model = MCMCModel(function=mcmc_wave, param_ranges=ranges, function_kwargs=kwargs)
    samples = model.metropolis_hastings(real_wave, NUM_ITERATIONS, NOISE_AMPLITUDE)
    noise = model.generate_noise(real_wave, noise_amp=NOISE_AMPLITUDE)

    for i, sample in enumerate(samples):
        current_params = sample
        acceptance_status = "Accepted"
        
        if i > 0:
            previous_params = samples[i - 1]
            
            if np.all(current_params == previous_params):
                acceptance_status = "Rejected (same as previous)"

        print(f"Trial {i+1}: ({acceptance_status})\n{current_params}")
    
    amplitude_samples = samples[:, 0]
    damping_samples = samples[:, 1]
    angular_freq_samples = samples[:, 2]
    # phase_samples = samples[:, 3]
    
    amplitude_samples = model.thin_samples(amplitude_samples, thin_ratio=THIN_RATIO, cutoff_start=int(NUM_ITERATIONS/10))
    damping_samples = model.thin_samples(damping_samples, thin_ratio=THIN_RATIO, cutoff_start=int(NUM_ITERATIONS/10))
    angular_freq_samples = model.thin_samples(angular_freq_samples, thin_ratio=THIN_RATIO, cutoff_start=int(NUM_ITERATIONS/10))
    
    mean_amplitude = np.mean(amplitude_samples)
    mean_damping = np.mean(damping_samples)
    mean_omega = np.mean(angular_freq_samples)
    
    result_wave = mcmc_wave(mean_amplitude, mean_damping, mean_omega, steps=DATA_STEPS)

    figure = plt.figure(figsize=(13, 7), layout="constrained")
    figure.suptitle("Extracting Signal Parameters From Noise")
    axes = figure.subplot_mosaic(
    """
    WWN.
    ABF.
    """,
    width_ratios=[1, 1, 1, 0.05]
    )
    
    axes["A"].hist(amplitude_samples, bins=100, density=True, label="estimated distribution")
    axes["A"].set_xlabel("Amplitude")
    ylim = axes["A"].get_ylim()
    axes["A"].plot([REAL_AMPLITUDE, REAL_AMPLITUDE], [0.0, ylim[1]], label="true value")
    axes["A"].plot([mean_amplitude, mean_amplitude], [0.0, ylim[1]], label="sample mean")
    axes["A"].legend(prop={"size": 8})
    
    axes["B"].hist(damping_samples, bins=100, density=True, label="estimated distribution")
    axes["B"].set_xlabel("Damping")
    ylim = axes["B"].get_ylim()
    axes["B"].plot([REAL_DAMPING, REAL_DAMPING], [0.0, ylim[1]], label = "true value")
    axes["B"].plot([mean_damping, mean_damping], [0.0, ylim[1]], label="sample mean")
    axes["B"].legend(prop={"size": 8})
    
    axes["F"].hist(angular_freq_samples, bins=100, density=True, label="estimated distribution")
    axes["F"].set_xlabel("Angular frequency")
    ylim = axes["F"].get_ylim()
    axes["F"].plot([REAL_OMEGA, REAL_OMEGA], [0.0, ylim[1]], label = "true value")
    axes["F"].plot([mean_omega, mean_omega], [0.0, ylim[1]], label="sample mean")
    axes["F"].legend(prop={"size": 8})
    
    # axes["P"].hist(phase_samples, bins=100, density=True, label="estimated distribution")
    # axes["P"].set_xlabel("phase constant")
    # axes["P"].plot([REAL_PHASE, REAL_PHASE], [0.0, 1.0], label = "true value")
    # axes["P"].legend()
        
    axes["W"].plot(time, real_wave, "r", label="real")
    axes["W"].set_xlabel("time")
    axes["W"].annotate(f"Real Parameters:\nAmplitude: {REAL_AMPLITUDE}\nDamping: {REAL_DAMPING}\nAngular Frequency: {REAL_OMEGA}", xy=(0.75, 0.1), xycoords="axes fraction")
    axes["W"].plot(time, result_wave, "g", label="generated")
    axes["W"].legend()
    
    axes["N"].plot(time, noise)
    axes["N"].set_xlabel("time")
    axes["N"].annotate(f"Noise amplitude: {NOISE_AMPLITUDE}", xy=(0.3, 0.9), xycoords="axes fraction")

    plt.show()
    
    
    