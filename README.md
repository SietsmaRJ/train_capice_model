# Train Capice Model
###### A placeholder repository till train model is implemented in the [CAPICE](https://github.com/molgenis/capice) repository.

# Requirements
Python 3.7+ (Does not work for python 3.6 or lower)

# Install
1. Clone or download the repository, forks are also viable.

2. Install the required libraries, either on the system or locally in a virtual environment:

    - On UNIX type systems:

        - For system wide install (requires admin access):
    `pip install -r requirements.txt`
    
        - For local virtual environment install:
    `bash venv_installer.sh`
    
    - On Windows type systems:
    
        - Please install the packages with their correct versions noted in 'requirements.txt'
       
# Usage
The program requires the following argument:
- -o/ --output: The output directory to put the new models and some datafiles in. Will make the directory if it does not exist.

One of either the following arguments is required:
- -f/ --file: The [CADD](https://cadd.gs.washington.edu/score) annotated data file to train a model on. Will make the data set balanced on the allele frequency and export that new dataset to output.
 If no balancing is desired, please use -b/ --balanced_ds for the input file.
- -b/ --balanced_ds: Same as -f/ --file, but will not balance out the allele frequency.

Optional arguments:
- -d/ --default: Use the XGBoost hyperparameters used by [Shuang et al.](https://www.medrxiv.org/content/10.1101/19012229v1.full.pdf) Will not use RandomizedSearchCV in the training phase.
- -s/ --split: Split the given input data into a train / test (0.8/0.2) set before any balancing, preprocessing or training happens. Will output both the training and testing dataset to output.
- -v/ --verbose: When added, will print out a whole lot of print messages for easier debugging.

# Output
Always:
- xgb_ransearch.pickle.dat: The pickled model instance. When argument -d/--default is used, will be a RandomizedSearchCV instance, instead of an XGBoostClassifier instance.

When argument -f/--file is used:
- train_balanced_dataset: GZipped export of the Allele Frequence balanced out training data file.

When argument -d/--default is __NOT__ used:
- xgb_optimal_model.pickle.dat: The best performing RandomizedSearchCV XGBoostClassifier model in a pickled instance.

When argument -s/--split is used:
- splitted_train_dataset: GZipped export of the split input data (80% of the data), used in training the model.
- splitted_test_dataset: GZipped export of the split (20% of the data). The model never sees this data, not even in evaluation.

Note:
If the -s/--split argument splits too aggressively, you may want to alter line 173's 'test_size' 0.2 value to the desired percentage (in 0-1).


# Why is this a placeholder?
The author of this repository, Robert J. Sietsma, is currently working on refactoring the [CAPICE](https://github.com/molgenis/capice) repository. The train_model.py script will be added to this repository Soon(TM).