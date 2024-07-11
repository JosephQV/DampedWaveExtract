import numpy as np
from wave_func import mcmc_wave

class MCMCModel:
    def __init__(self, function, param_ranges: np.ndarray, function_kwargs: dict = None):
        self.function = function
        self.param_ranges = param_ranges
        self.function_kwargs = function_kwargs
        self.rng = np.random.default_rng(seed=824)
    
    def generate_noise(self, data, scale, noise_amp=1.0):
        noise = 2 * noise_amp * self.rng.normal(loc=0.0, scale=scale, size=len(data)) - noise_amp
        return data + noise

    def compute_rms(self, observed, predicted):
        diff = observed - predicted
        return np.sqrt(np.mean(diff**2))

    def guess_params(self, distribution=None):
        if distribution == "norm":
            mu = (self.param_ranges[:,0] + self.param_ranges[:,1]) / 2
            sigma = mu/10
            return self.rng.normal(mu, sigma, size=len(self.param_ranges))
        return self.rng.uniform(low=self.param_ranges[:,0], high=self.param_ranges[:,1], size=len(self.param_ranges))
        
    def likelihood(self, noisy_vals, params, tol=1e-9):
        trial_vals = eval(self.function.__name__)(*params, **self.function_kwargs)
        rms_error = self.compute_rms(noisy_vals, trial_vals)
        likelihood = 1 / (rms_error + tol) 
        return likelihood

    def metropolis_hastings(self, data, num_iterations, noise_scale, noise_amplitude=1.0):
        noisy_vals = self.generate_noise(data, scale=noise_scale, noise_amp=noise_amplitude)

        current_params = self.guess_params()

        current_likelihood = self.likelihood(noisy_vals, current_params)

        samples = np.empty(shape=(num_iterations, len(self.param_ranges)))
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
    
    def thin_samples(self, samples, thin_percentage: float = 0.25, cutoff_percentage: float = 0.2):
        # Thin percentage: percentage of values from samples to keep after thinning, 0.25 keeps 25% of the values
        # cutoff percentage: percentage of values from the beginning of samples to skip, 0.2 would start at 20% through the array
        if thin_percentage == 0:
            return np.array([])
        start = int(samples.size * cutoff_percentage)
        count = int(thin_percentage * (samples.size - start))
        indices = self.rng.integers(start, samples.size, count)
        thinned = samples[indices]
        return thinned
    
    def print_sample_statuses(self, samples):
        for i, sample in enumerate(samples):
            current_params = sample
            acceptance_status = "Accepted"
            
            if i > 0:
                previous_params = samples[i - 1]
                
                if np.all(current_params == previous_params):
                    acceptance_status = "Rejected (same as previous)"

            print(f"Trial {i+1}: ({acceptance_status})\n{current_params}")
    

if __name__ == "__main__":
    model = MCMCModel(function=mcmc_wave, param_ranges=np.arange(3), function_kwargs={"steps":5, "seconds": 12})
    print(model.likelihood(np.arange(5), [400, 400, 400]))
    
    samples = np.arange(100)
    print(np.sort(model.thin_samples(samples)))
    
    
    
    