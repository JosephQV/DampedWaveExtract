Notes
- Model is used on waves with noise of amplitude 0, 1.5, and 4 for the 3 figure images in the repo. 1 million iterations were performed for the sampling of each of these. Accuracy of the parameter estimation quickly drops with noise amplitude
that is about 25% as much as the real wave's amplitude. See the plot of the accuracy measured using RMS versus increasing noise amplitude.
- Samples for each parameter returned by metropolis_hastings are thinned by eliminating the first 50% of values and only using 50% of the remaining values. The thinning has a
minor effect on improving the parameter estimations.
