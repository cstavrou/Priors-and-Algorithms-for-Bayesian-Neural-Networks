# Chapter 2

The code in this section was used to draw all the prior samples and create the plots that were used for Chapter 2 of the dissertation.

**Options:**
  1.  Activation function (str: tanh, relu or softmax)
  2.  Prior distribution for the parameters (str: T, laplace, normal, cauchy, mix or spike_slab) 
  3.  Number of hidden units for each hidden layer (int)
  4.  Degrees of freedom for the T distribution (float > 0)
  5.  Number of samples of x1 and x2 (int > 0)
  6.  Graph limit (float > 0)
  7.  Number of hidden layers (int = 0, 1 or 2)            				     
  8.  Prior dispersion parameter (float > 0) [note this is then rescaled accordingly]
        1. Normal: sigma = std/sqrt(d_j), where d_j are units from previous layer
        2. Laplace: b = std^2/d_j
        3. StudentT: s = std^2/d_j
# 9.  seed                        			   (random seed)
# 10. n_iter                 				   (number of iterations for resampling prior)
# 11. T/F                                      (whether the first layer will have std for priors)
# 12. prob                   				   (probability for spike_slab prior) [ONLY if prior = spike_slab]
# 13. prior1                  				   (first prior distribution)  [ONLY if prior = mix]  
# 14. prior2                   				   (second prior distribution) [ONLY if prior = mix]
