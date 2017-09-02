# CO2 Regression:
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

**Options:**
  1. Minibatch size (must be an integer dividing 848 exactly)
  2. Number of hidden units in each hidden layer (int)
  3. Number of hidden layers (0 < int < 3)
  4. Activation function (str: relu, tanh, softplus)
  5. Optimiser (str: sgd/adam/momentum/nag)
  6. Number of epochs (int)
  7. Learning rate (0 < float < 1)
  
  
  
