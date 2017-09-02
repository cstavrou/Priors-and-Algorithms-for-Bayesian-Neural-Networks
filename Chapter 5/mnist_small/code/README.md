# Small MNIST (classification):
This code used the *co2_regression* dataset described in the dissertation. 

All the code files need to be in a folder named **code**, and the data in a folder named **data**. In order for the code to work we require that the structure of the *co2_regression* folder is kept. 

All scripts can be ran using the **terminal**, by first navigating to the *code* folder.

Once you have navigated to the right folder you can run each script as follows:


## 1. make_data_files.py

This script makes the specific **training** and **test** datasets that are used in the dissertation. Please note that they are already in the **data** folder. 

To run the files type the following command in the terminal:
  1. [LINUX:]
  python make_data_files.py
  2. [Windows:]
  py.exe make_data_files.py
  

## 2. co2_bp.py

This script fits feed-forward neural networks using the *co2_regression* dataset. The script allows the user to input 7 options when running this script. The script allows for models with 1 or 2 hidden layers using any number of hidden units. The minibatch size is restrictive to integers dividing 848 without remainder. Three possible activation functions can be used relu, tanh and softplus. The user can pick one of the 4 different optimisers sgd, adam, momentum or nag (Nesterov's accelerated gradient). The number of epochs must be an integer greater than 0. The learning rate for the optimisation step can also be defined. Please note that for the adam optimiser all other options apart from the learning rate are left to their default values.

When the script is ran a new folder path "saved/... model specifics ..." will be created where the plots used for the feed-forward neural networks will be saved along with the information file *info_file.csv*, which contains the summary of the specific model, e.g. running times, batch size, and the final MSE.

**Options:**
  1. Minibatch size (must be an integer dividing 848 exactly)
  2. Number of hidden units in each hidden layer (int)
  3. Number of hidden layers (int: 1 or 2)
  4. Activation function (str: relu, tanh, softplus)
  5. Optimiser (str: sgd/adam/momentum/nag)
  6. Number of epochs (int)
  7. Learning rate (0 < float < 1)
  
 **Example:** Suppose that you want to run a model with batch size of 212, 500 hidden units, 2 hidden layers, using softplus activation functions, the sgd optimiser, 5000 epochs and a learning rate of 0.001. Then in terminal you would run the following command,
 
  1. [LINUX:]
  python co2_bp.py 212 500 2 softplus sgd 5000 0.001
  2. [Windows:]
  py.exe co2_bp.py 212 500 2 softplus sgd 5000 0.001
  
  
## 3. co2_regression_1l.py

This script performs HMC or SGHMC inference on a 1 hidden layer Bayesian neural network (BNN). The inference is performed sequentially to allow for computers with smaller available memory. 

Two phases are performed: 
  Phase 1: (initial burnin period) the first chunk of samples ran are discarded as an initial burnin. The burnin can be manually adjusted after the script has finished running and the sample files have been saved using the *make_plots.py* script.
  Phase 2: (sampling phase) this sampling phase is performed iteratively, for example one could have 10 phase iterations of 1000 samples, meaning that a total of 10 x 1000 = 10000 will be collected. Note that an additional 1000 samples would have been burnt in during Phase 1. 
  
The reason for this split is in order to benefit from the gpu computation whilst allowing computers with smaller available memory to run the code.

Again, when the script is ran a new folder path "saved/... model specifics ..." will be created where the plots used for the BNNs using HMC or SGHMC will be saved along with the information file *info_file.csv*, which contains the summary of the specific model, e.g. running times, batch size, and the final MSE, and so on. Many output files and the samples collected will be saved in that folder (for the larger models please ensure that there is sufficient disk space). None of the files created must be deleted or renamed, as they are used to create additional plots used in the dissertation using the *make_plots.py* script.
  
**Options:**
  1. discount factor
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
  11. Whether the file is to be ran on AWS (str: True or False) 
    if True: no plots are produced
    if False: produces plots
  12. (Only if prior = T) The degrees of freedom for the T distribution (float > 0) 
  
The discount factor should be set to 0, unless you want the leapfrog step size to increase/decrease for the first 2/3 phase iterations and decrease/increase in the last 1/3 phase iterations.

The number of burned samples at the beginning of each new iteration of phase 2 should be set equal to 1, unless otherwise required.
 
step_size_new = step_size * (1 - discount factor)^n, for n < int(2/3 * no. of phase iterations) = m

step_size_new = step_size * (1 - discount factor)^m * (1 + discount factor)^(k - m), for k >= int(2/3 * no. of phase iterations)
 
 **Example:** Suppose that you want to run a model with discount factor = 0, 500 hidden units, using hmc, normal prior, 1000 samples per phase iteration, 0.001 leapfrog step size, 70 leapfrog steps, 1 burned samples at the beginning of each new iteration, dispersion parameter of 15, 10 phase 2 iterations, and producing graphics. Then in terminal you would run the following command,
 
  1. [LINUX:]
  python co2_regression_1l.py 0 500 hmc normal 1000 0.001 70 1 15 10 False
  2. [Windows:]
  py.exe co2_regression_1l.py 0 500 hmc normal 1000 0.001 70 1 15 10 False
  
  
## 4. co2_regression_2l.py

This takes the same commands as the **co2_regression_1l.py** and performs HMC or SGHMC inference on a BNN with 2 hidden layers.


## 5. co2_regression_1l_VI.py

Performs Variational Inference (VI) on a 1 hidden layer Bayesian NN, using the Variational Mean Field (VMF) approximating distribution. 

Again, when the script is ran a new folder path "saved/... model specifics ..." will be created where the plots used for the BNNs using VI will be saved along with the information file *info_file.csv*, which contains the summary of the specific model, e.g. running times, batch size, and the final MSE, and so on.

**Options:**
  1. Number of hidden units (int)
  2. Number of variational inference iterations (int)
  3. Prior dispersion parameter (float > 0) [note this is then rescaled accordingly]
      1. Normal: sigma = std/sqrt(d_j), where d_j are units from previous layer 
      2. Laplace: b = std^2/d_j 
      3. StudentT: s = std^2/d_j
  4. Whether the approximating distribution has non_zero_mean (str: True or False)
      1. If True: approximating distribution has a non-zero mean
      2. If False: then the approximating distribution has a zero mean
  5. Prior distribution (str: normal, laplace or T)
  6. (Only if prior = T) The degrees of freedom for the T distribution (float > 0) 
 
 
## 6. co2_regression_pymc3_1l.py

Performs Automatic Differentation Variational Inference (ADVI) on a 1 hidden layer Bayesian NN. No plots or results are stored, instead the final Monte Carlo estimate MSE is printed in the terminal. This was not explored further as it was deviating from the main topic of the dissertation.

**Options:**
  1. Number of hidden units (int)
  2. Number of variational inference iterations (int)
  3. Prior distribution (str: normal or laplace)
  4. Prior dispersion parameter (float > 0) [note this is then rescaled accordingly]
      1. Normal: sigma = std/sqrt(d_j), where d_j are units from previous layer 
      2. Laplace: b = std^2/d_j 
  5. Output standard deviation for regression likelihood (float > 0)


## 7. make_plots.py

The script to make all the required plots for the models using HMC or SGHMC inference. This script must be ran after the inference is ran and all the files required to make the plots and additional statistics have been stored in the *"/saved/...."* directory. This script requires that the location, and the name of all the files created from the inference remain unchanged.

All the plots are saved in the corresponding folder, please note that in the terminal the Monte Carlo test MSE estimate and its standard deviation are printed, along with the number of samples collected and the point estimate test MSE (these are not stored anywhere else, therefore might need to be copied from the terminal).

**Options:**
  1. Number of hidden units (int)
  2. Inference method (str: hmc or sghmc)
  3. Prior distribution (str: normal, laplace or T)
  4. Total number of samples (int)
      * This corresponds to the number of samples that can be found in the name of the folder name (or instead no. phase iterations x number of samples per phase iteration)
  5. Number of hidden layers (str: 1l or 2l)   
  6. The last file number, equivalently this is no. phase iterations - 1 (int)
  7. Whether to make a trace plot for the weights (str: True or False)
  8. Whether to make the marginal distributions plots (str: True or False)
  9. Whether to skip the burnin phase (phase 1) when plotting the MSE trace (str: True or False)
  10. Whether to use the additional per 100 samples collected (str: True or False)
      * These are samples stored once every 100 samples during the sampling stage, we can then space these out manually using the commands for options 12 and 13. Allows a manual increase of the burnin and increase of sample spacing
  11. Whether to show all posterior samples on predictive plot  (str: True or False)
  12. Every how many samples from the samples of 1 per 100 to pick a sample (int)
      * Note that using 2 for example means that we take 1 sample every 200, 3 means 1 sample every 300 and so on
  13. Proportion of the first samples that are burnt (0 < float < 1)
  14. (Only if prior = T) The degrees of freedom for the T distribution (float > 0) 




