Code to run the simulations in the paper Intelligent Pooling at RL for Real Life ICML workshop.

Dependencies: This code is all run in the anaconda python 3.6 environment. Please download this environment to run the appropriate code. [anconda environments link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

You will need numpy and pandas. After you install and activate py36, you can then update pandas and numpy. 

The Gaussian process model is implemented in GPyTorch. You can find that here: [GPyTorch](https://gpytorch.ai/).

To run code see ```run_multiple_32_7.sh```. The options for this script are as follow: 

* The number of participants, the current option is 32. 
* The time at which to perform updates, reported results have this set to 7. 
* Simulation start index, currently set to 0. 
* Simulation end index, currently set to 1. 
* The training method, the current option is EB for empirical Bayes. 
* The approach option, current set to pooling. Other options are: 
batch or personalized.

