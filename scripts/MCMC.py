import numpy as np
from wave_funcs import *
from utility_funcs import generate_noise, compute_rms, guess_params, evaluate_wave_fcn


class MCMCModel:
    def __init__(self, wave_fcn, param_ranges: np.ndarray, wave_kwargs: dict = None):
        self.wave_fcn = wave_fcn
        self.param_ranges = param_ranges
        self.wave_kwargs = wave_kwargs
        
    def likelihood(self, noisy_vals: np.ndarray, params: np.ndarray, tol=1e-9):
        trial_vals = evaluate_wave_fcn(self.wave_fcn, params, self.wave_kwargs)
        rms_error = compute_rms(noisy_vals, trial_vals)
        likelihood = 1 / (rms_error + tol) 
        return likelihood

    def metropolis_hastings(self, data: np.ndarray, num_iterations: int, snr: float) -> np.ndarray:
        noisy_vals = generate_noise(data, snr=snr)

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
    
    model = MCMCModel(wave_fcn=damped_wave, param_ranges=np.random.randint(0, 10, 6).reshape(3,2), wave_kwargs={})
    
    real_wave = damped_wave([8.0, 0.5, 1.0])
    samples = model.metropolis_hastings(data=real_wave, num_iterations=100, snr=1.0)
    
    print_sample_statuses(samples)
    
    
    
    
    
    