# Class Incremental Learning Using Enhanced Replay and Bias Correction Using Auxiliary Network
In this work, we propose a novel exemplar selection algorithm called **CORE-high** that is effective when dataset consists of noisy images. This method outperforms plain random selection when the images contain random pixel noise. We also propose a bias correction technique that makes use of an auxiliary network **BiC-AN** which mitigates the biases against old classes. 

### Installation & Requirements

The code is tested on Python 3.9.6 using PyTorch 1.9.0 and CUDA 11.1. We provide ```requirements.txt``` file that can be used to install the important packages using pip:
```
pip install -r requirements.txt
```

### Datasets
* <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">CIFAR100</a>
* <a href="http://cs231n.stanford.edu/reports/2016/pdfs/401_Report.pdf" target="_blank">TinyImageNet</a>

We make use of the <a href='https://github.com/Continvvm/continuum' target="_blank">Continuum</a> library to load and split the datasets into various tasks.  

## Experiments
For each experiment, the models are saved after every epoch using a filename inside the folder ```store```. The saved models are automatically loaded from the previous checkpoint if the experiment is paused and resumed at a later time.

### Running Replay Experiments:
The ```replay.sh``` can be used to run replay experiments. The variables ```permstart```, ```permend``` can be modified to set the seed that generates class orders. The variables ```startseed``` and ```repeat``` can be used to set the start seed and the number or runs for each class order. By default the three methods that are compared are: ```CORE-high, random and herding``` The AGENTS variable in ```main.py``` can be used to specify the replay methods to compare. 

### Running Bias Correction Experiments
The ```bic.sh``` can be used to run the bias correction experiments.

### Running Custom Experiments
Individual experiments can be run using ```main.py```. The ```get_args(..)``` method in ```utils.py``` shows all available command line arguments. Some of the main arguments are:
- ```--dataset``` (CIFAR100 (default) | TinyImageNet200)
- ```--repeat```: number of repeats
- ```--initial_increment```: number of base classes
- ```--increment```: number of classes per task for incremental classes
- ```--memory_type```: fixed vs flexible memory
- ```--memory_per_class```: number of samples for each incremental class
- ```--noise_var``` : noise variance parameter
-``` --perm``` : class order number
```--output_to_file```: if results need to be output, otherwise just runs a training session of the models 
- ```--extra_info```: extra information to be added to the folder containing experiment results
-```--correct_bias```: whether bias correction needs to be applied
- ```--bic_method```: the bias correction method to be used

## Acknowledgements
- [1] M. Masana, X. Liu, B. Twardowski, M. Menta, A. D. Bagdanov, and J. van de Weijer, “Class-incremental learning: Survey and performance evaluation on image 
classification”, arXiv preprint arXiv:2010.15277, 2020.
- [2] A. Douillard and T. Lesort, “Continuum: Simple management of complex continual learning  scenarios”,  arXiv  preprint  arXiv:2102.06253, 2021.
- [3] Hsu, Y. C., Liu, Y. C., Ramasamy, A., & Kira, Z. (2018). Re-evaluating continual learning scenarios: A categorization and case for strong baselines. arXiv preprint arXiv:1810.12488.
- [4] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. the Journal of machine Learning research, 12, 2825-2830.







