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
            if acceptance_ratio > np.random.uniform():
                current_params = proposed_params
                current_likelihood = proposed_likelihood

            samples[i] = current_params

        return samples
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from wave_func import wave, mcmc_wave
    REAL_AMPLITUDE = 6.0
    REAL_DAMPING = 0.3
    REAL_OMEGA = 2.0
    REAL_PHASE = 0.0
    NOISE_AMPLITUDE = 0.0
    NUM_ITERATIONS = 10000
    
    time, real_wave = wave(REAL_AMPLITUDE, REAL_DAMPING, REAL_OMEGA, phase=REAL_PHASE)
    
    ranges = np.array(
        [
            [0.1, 10.0],    # amplitude (A) range
            # [0.1, 5.0],     # damping (b) range
            # [0.1, 50.0],    # angular frequency (omega) range
            # [0.0, 2*np.pi]  # phase constant range
        ]
    )
    kwargs = {"seconds": 30.0, "steps": 1000}

    model = MCMCModel(function=mcmc_wave, param_ranges=ranges, function_kwargs=kwargs)
    samples = model.metropolis_hastings(real_wave, NUM_ITERATIONS, NOISE_AMPLITUDE)

    for i, sample in enumerate(samples):
        current_params = sample
        acceptance_status = "Accepted"
        
        if i > 0:
            previous_params = samples[i - 1]
            
            if np.all(current_params == previous_params):
                acceptance_status = "Rejected (same as previous)"

        print(f"Trial {i+1}: ({acceptance_status})\n{current_params}")
    
    amplitude_samples = samples[:, 0]
    # damping_samples = samples[:, 1]
    # angular_freq_samples = samples[:, 2]
    # phase_samples = samples[:, 3]
    
    axes = plt.figure(layout="constrained").subplot_mosaic(
    """
    WN
    AA
    """
    )
    
    axes["W"].plot(time, real_wave)
    axes["W"].annotate(f"A: {REAL_AMPLITUDE}  b: {REAL_DAMPING}  omega: {REAL_OMEGA}  phase: {REAL_PHASE}", xy=(0.6, 0.9), xycoords="axes fraction")
    
    axes["A"].hist(amplitude_samples, bins=100, density=True, label="estimated distribution")
    axes["A"].set_xlabel("amplitude")
    axes["A"].plot([REAL_AMPLITUDE, REAL_AMPLITUDE], [0.0, 1.0], label = "true value")
    axes["A"].legend()
    
    # axes["B"].hist(damping_samples, bins=100, density=True, label="estimated distribution")
    # axes["B"].set_xlabel("damping")
    # axes["B"].plot([REAL_DAMPING, REAL_DAMPING], [0.0, 1.0], label = "true value")
    # axes["B"].legend()
    
    # axes["F"].hist(angular_freq_samples, bins=100, density=True, label="estimated distribution")
    # axes["F"].set_xlabel("angular frequency (omega)")
    # axes["F"].plot([REAL_OMEGA, REAL_OMEGA], [0.0, 1.0], label = "true value")
    # axes["F"].legend()
    
    # axes["P"].hist(phase_samples, bins=100, density=True, label="estimated distribution")
    # axes["P"].set_xlabel("phase constant")
    # axes["P"].plot([REAL_PHASE, REAL_PHASE], [0.0, 1.0], label = "true value")
    # axes["P"].legend()
    
    
    plt.show()
    # samples_array = np.array(samples)
    # v_samples = samples_array[:,1]
    # a_samples = samples_array[:,0]
    
    # plt.hist(v_samples, 100, density=True, label = "estimated distribution")
    # plt.xlabel("velocity")
    # ylim = plt.ylim()
    # yx =plt.gca()
    # plt.plot([-1,-1], ylim, label = "true value")
    # yx.set_ylim(ylim)
    # plt.legend()
    
    
    # Pass an array containing any number of parameters that need to be sampled
    
    # Pass data that the algorithm will compare against instead of calculating that data itself
    
    
    