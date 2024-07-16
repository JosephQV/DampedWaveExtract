import numpy as np
from wave_funcs import mcmc_wave
from utility_funcs import generate_noise, compute_rms, guess_params


class MCMCModel:
    def __init__(self, function, param_ranges: np.ndarray, function_kwargs: dict = None):
        self.function = function
        self.param_ranges = param_ranges
        self.function_kwargs = function_kwargs
        
    def likelihood(self, noisy_vals, params, tol=1e-9):
        trial_vals = eval(self.function.__name__)(*params, **self.function_kwargs)
        rms_error = compute_rms(noisy_vals, trial_vals)
        likelihood = 1 / (rms_error + tol) 
        return likelihood

    def metropolis_hastings(self, data, num_iterations, noise_scale, noise_amplitude=1.0):
        noisy_vals = generate_noise(data, scale=noise_scale, noise_amp=noise_amplitude)

        current_params = guess_params(param_ranges=self.param_ranges)

        current_likelihood = self.likelihood(noisy_vals, current_params)

        samples = np.empty(shape=(num_iterations, len(self.param_ranges)))
        samples[0] = current_params
        
        for i in range(1, num_iterations):
            proposed_params = guess_params(param_ranges=self.param_ranges)
            proposed_likelihood = self.likelihood(noisy_vals, proposed_params)

            acceptance_ratio = proposed_likelihood / current_likelihood
            if acceptance_ratio > np.random.uniform(0.7, 1.0):
                current_params = proposed_params
                current_likelihood = proposed_likelihood

            samples[i] = current_params

        return samples
    

if __name__ == "__main__":
    from utility_funcs import print_sample_statuses
    model = MCMCModel(function=mcmc_wave, param_ranges=np.arange(6).reshape(3,2), function_kwargs={})
    wave = mcmc_wave(5, 0.5, 2)
    samples = model.metropolis_hastings(wave, 100, 2)
    print_sample_statuses(samples)
    
    
    
    
    
    