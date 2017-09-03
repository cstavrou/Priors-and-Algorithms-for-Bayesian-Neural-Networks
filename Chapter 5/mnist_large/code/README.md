# Large MNIST dataset

The code in this section is all the supporting code for the dissertation regarding the large MNIST datasets, along with additional code where the results were not presented in the thesis due to space limitations, and the time required to run the code.

This code used the *(large) MNIST* dataset described in the dissertation. 

All the code files need to be in a folder named **code**, and the data in a folder named **data**. In order for the code to work we require that the structure of the *mnist_large* folder is kept. 

All scripts can be ran using the **terminal**, by first navigating to the *code* folder.

Once you have navigated to the right folder you can run each script as follows:
   
## 4. mnist_1l_large.py
This script performs HMC or SGHMC inference on a 1 hidden layer Bayesian neural network (BNN). The inference is performed sequentially to allow for computers with smaller available memory. 

Two phases are performed: 
  Phase 1: (initial burnin period) the first chunk of samples ran are discarded as an initial burnin. The burnin can be manually adjusted after the script has finished running and the sample files have been saved using the *make_plots.py* script.
  Phase 2: (sampling phase) this sampling phase is performed iteratively, for example one could have 10 phase iterations of 1000 samples, meaning that a total of 10 x 1000 = 10000 will be collected. Note that an additional 1000 samples would have been burnt in during Phase 1. 
  
The reason for this split is in order to benefit from the gpu computation whilst allowing computers with smaller available memory to run the code.

Again, when the script is ran a new folder path "saved/... model specifics ..." will be created where the plots used for the BNNs using HMC or SGHMC will be saved along with the information file *info_file.csv*, which contains the summary of the specific model, e.g. running times, batch size, and the final test prediction accuracy, and so on. Many output files and the samples collected will be saved in that folder (for the larger models please ensure that there is sufficient disk space). None of the files created must be deleted or renamed, as they are used to create additional plots used in the dissertation using the *make_plots.py* script.
  
**Options:**
  1. Batch size (int < 55000)
  2. Number of hidden units (int)
  3. Method of inference (str: hmc or sghmc)
  4. Prior distribution (str: normal, laplace or T)
  5. Number of samples per phase iteration (int)
  6. Leapfrog step size (0 < float < 1)
  7. Number of leap frog steps (int)
  8. Number of burned samples at the beginning of each new iteration of phase 2
  9. Prior dispersion parameter (float > 0) [note this is then rescaled accordingly]
      1. Normal: sigma = std/sqrt(d_j), where d_j are units from previous layer 
      2. Laplace: b = std^2/d_j 
      3. StudentT: s = std^2/d_j
  10. Number of phase 2 iterations (int > 0)
  11. Discount factor
  12. (Only if prior = T) The degrees of freedom for the T distribution (float > 0) 
  
The discount factor should be set to 0, unless you want the leapfrog step size to increase/decrease for the first 2/3 phase iterations and decrease/increase in the last 1/3 phase iterations.

The number of burned samples at the beginning of each new iteration of phase 2 should be set equal to 1, unless otherwise required.
 
step_size_new = step_size * (1 - discount factor)^n, for n < int(2/3 * no. of phase iterations) = m

step_size_new = step_size * (1 - discount factor)^m * (1 + discount factor)^(k - m), for k >= int(2/3 * no. of phase iterations)

*Note that ideally, we would want the batch size to be 55000 (number of training points) however as the images are too large to have in memory and to perform all the required computations we typically used a batch size of 10000. You can always try using a batch of 55000 to see whether your computer can handle it*

 **Example:** Suppose that you want to run a model with batch size = 10000, 500 hidden units, using hmc, T prior, 1000 samples per phase iteration, 0.001 leapfrog step size, 70 leapfrog steps, 1 burned samples at the beginning of each new iteration, dispersion parameter of 15, 10 phase 2 iterations, a discount factor = 0, 2.5 degrees of freedom for T distribution. Then in terminal you would run the following command,
 
  1. [LINUX:]
  python mnist_1l_large.py 10000 500 hmc T 1000 0.001 70 1 15 10 2.5 0
  2. [Windows:]
  py.exe mnist_1l_large.py 10000 500 hmc T 1000 0.001 70 1 15 10 2.5 0
  
  
## 2. mnist_2l_large.py

This takes the same commands as the **mnist_1l_large.py** and performs HMC or SGHMC inference on a BNN with 2 hidden layers. *(please note that this file does not collect the probability trace like *mnist_1l_large.py* this will a future addition)*

## 3. mnist_conv_bnn_large.py

This takes the same commands as the **mnist_1l_large.py** and performs HMC or SGHMC inference on a Bayesian convolutional neural network. *(please note that this file does not collect the probability trace like *mnist_1l_large.py* this will a future addition)*

## 4. make_plots.py

The script to make all the required plots for the models using HMC or SGHMC inference. This script must be ran after the inference is ran and all the files required to make the plots and additional statistics have been stored in the *"/saved/...."* directory. This script requires that the location, and the name of all the files created from the inference remain unchanged.

All the plots are saved in the corresponding folder, please note that in the terminal the Monte Carlo test prediction accuracy estimate and its standard deviation are printed, along with the number of samples collected and the point estimate test prediction accuracy (these are not stored anywhere else, therefore might need to be copied from the terminal).

**Options:**
  1. Number of hidden units (int)
  2. Inference method (str: hmc or sghmc)
  3. Prior distribution (str: normal, laplace or T)
  4. Total number of samples (int)
      * This corresponds to the number of samples that can be found in the name of the folder name (or instead no. phase iterations x number of samples per phase iteration)
  5. Model (str: 1l, 2l or conv_net)   
  6. The last file number, equivalently this is no. phase iterations - 1 (int)
  7. Whether to make a trace plot for the weights (str: True or False)
  8. Whether to make the probability trace plot (str: True or False)
  9. Whether to make the marginal distributions plots (str: True or False)
  10. Whether to skip the burnin phase (phase 1) when plotting the prediction accuracy trace (str: True or False)
  11. Whether to use the additional per 100 samples collected (str: True or False)
      * These are samples stored once every 100 samples during the sampling stage, we can then space these out manually using the commands for options 12 and 13. Allows a manual increase of the burnin and increase of sample spacing
  12. Every how many samples from the samples of 1 per 100 to pick a sample (int)
      * Note that using 2 for example means that we take 1 sample every 200, 3 means 1 sample every 300 and so on
  13. Proportion of the first samples that are burnt (0 < float < 1)
  14. (Only if prior = T) The degrees of freedom for the T distribution (float > 0) 

*Please note that for the 2l and conv_net models argument 8 must be set to* **False** *as the probability trace plots are not created by the code (this will be changed in the future)*.
