# Train Capice Model
###### A placeholder repository till train model is implemented in the [CAPICE](https://github.com/molgenis/capice) repository.

This tool is developed within PyCharm 2020.2 on an UNIX type system. Performance on other systems is not guaranteed.

# Requirements
Python 3.7+ 

Performance on python 3.6 or lower is not tested.

The following Python packages are required:
 * numpy ([v1.19.2](https://github.com/numpy/numpy/releases); [BSD 3-Clause License](https://www.numpy.org/license.html))
 * pandas ([v1.1.0](https://github.com/pandas-dev/pandas); [BSD 3-Clause License](https://github.com/pandas-dev/pandas/blob/master/LICENSE))
 * scipy ([v1.5.2](https://github.com/scipy/scipy); [BSD 3-Clause License](https://github.com/giampaolo/psutil/blob/master/LICENSE))
 * scikit-learn ([v0.23.2](https://scikit-learn.org/stable/whats_new.html); [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/master/COPYING))
 * xgboost ([v1.1.1](https://github.com/dmlc/xgboost); [Apache 2 License](https://github.com/dmlc/xgboost/blob/master/LICENSE))

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
- -d/ --default: Use the XGBoost hyperparameters used by [Shuang et al.](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-020-00775-w) Will not use RandomizedSearchCV in the training phase.
- -sd/ --specified_default: Location to a json containing keys: `learning_rate`, `max_depth`, `n_estimators` and their values to be used as the default hyperparameters in training the model.
- -s/ --split: Splits the given data into a given test set of the given percentage size and train datasets before any preprocessing or training happens. Will output both the training and testing dataset to output. Expects a percentage from 0-1.
- -v/ --verbose: When added, will print out a whole lot of print messages for easier debugging.
- -e/ --exit: Early exit flag, when the given input data should just be split or balanced out. Will exit before any preprocessing or training happens.

### Example:

- Balance out the given file and make cross validation models on the balanced set:
    - `python3 train_model.py -f path/to/cadd/annotated/file -o path/to/output -d -v` 

- Do not balance out the given dataset and split it to a 90/10 train/test dataset:
    - `python3 train_model.py -b path/to/balanced/cadd/annotated/file -o path/to/output -v -s 0.1`

- Make a model using previously found optimal hyperparameters, without balancing out the input dataset:
    - `python3 train_model.py -b path/to/cadd/annotated/file -o path/to/output -v -s 0.1 -d -sd path/to/hyperparameters.json`
    
- Balance out and split a dataset, without training:
    - `python3 train_model.py -f path/to/cadd/annotated/file -o path/to/output -s 0.1 -v -e`

# Output
Always:
- xgb_ransearch.pickle.dat: The pickled model instance. When argument -d/--default is used, will be a RandomizedSearchCV instance, instead of an XGBoostClassifier instance.

When argument -f/--file is used:
- train_balanced_dataset: GZipped export of the Allele Frequence balanced out training data file.

When argument -d/--default is __NOT__ used:
- xgb_optimal_model.pickle.dat: The best performing RandomizedSearchCV XGBoostClassifier model in a pickled instance.

When argument -s/--split is used:
- splitted_train_dataset: GZipped export of the split input data, used in training the model.
- splitted_test_dataset: GZipped export of the split. The model never sees this data, not even in evaluation.

# Why is this a placeholder?
The author of this repository, Robert J. Sietsma, is currently working on refactoring the [CAPICE](https://github.com/molgenis/capice) repository. The train_model.py script will be added to this repository Soon(TM).

#TODO
- Make impute_preprocess more future proof by adding classes of CADD values and impute values.
- Make -sd also skip ransearch.

#FAQ
- Q: My model training failed with an error in _joblib.externals.loky.process_executor.RemoteTraceback_ with "ValueError: unknown format is not supported". Why?
    - A: This is possibly because the training data size is not large enough for a RandomSearchCV. Please try to increase the training data size or use the -b flag for the input.