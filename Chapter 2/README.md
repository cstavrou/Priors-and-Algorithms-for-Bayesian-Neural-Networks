# Chapter 2

The code in this section was used to draw all the prior samples and create the plots that were used for Chapter 2 of the dissertation.

**Options:**
  1.  Activation function (str: tanh, relu or softmax)
  2.  Prior distribution for the parameters (str: T, laplace, normal, cauchy, mix or spike_slab) 
  3.  Number of hidden units for each hidden layer (int)
  4.  Degrees of freedom for the T distribution (float > 0)
  5.  Number of samples of x1 and x2 (int > 0)
      * We then make a grid from the x1 and x2, so putting 100 here -> 100^2 grid combinations.
  6.  Graph limit (float > 0)
  7.  Number of hidden layers (int = 0, 1 or 2)            				     
  8.  Prior dispersion parameter (float > 0) [note this is then rescaled accordingly]
        1. Normal: sigma = std/sqrt(d_j), where d_j are units from previous layer
        2. Laplace: b = std^2/d_j
        3. StudentT: s = std^2/d_j
  9.  Random seed (int)
  10. Number of iterations for resampling prior (int)
  11. Whether the first layer will use dispersion parameter defined for the priors (str: T or F)
      * If T: the first layer prior dispersion parameter is computed like shown argument 8
      * If F: the first layer prior dispersion parameter is set to 1, regardless the distribution
  12. [ONLY if prior = spike_slab] Probability for spike_slab prior (0 < float < 1) 
  13. [ONLY if prior = mix]  First prior distribution (str: T, laplace, normal or cauchy)  
  14. [ONLY if prior = mix] Second prior distribution (str: T, laplace, normal or cauchy)
  
  
  Please note that even if the degrees of freedom argument is **not** required as a different distribution than a T distribution is being used, we still need to put a value in that position of the arguments so that the program can read the rest of the arguments correctly. For the *mix* prior, we cannot define a combination of 2 different T distribution priors as we only have one degrees of freedom argument. 
  
The **mix** prior can only be enabled for the 2 hidden layer networks. If the mix prior is being used then the first two layers of parameters (parameters connecting the inputs to the first hidden layer and the parameters connecting the first hidden layer to the second hidden layer) have the distribution defined by the first prior distribution (argument 13), and the last layer of parameters have the distribution defined by the second prior distribution (argument 14).

Please note, that for any arguments not required you should still input a value (which is not actually used) as the arguments are assigned to the correct variable based on their position within the input argument list.
  
  
