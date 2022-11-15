# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Udacity project to train a model that predicts customer churn for a bank.

## Files and data description
```
.
├── environment.yml                    # Conda environment
├── churn_library.py                   # Scrpt to train models on bank data
├── churn_script_logging_and_tests.py  # Testing and logging our script
├── README.md           
├── data                 
│   └── bank_data.csv                  # Data to train/test model
├── images                             # EDA results 
│   ├── eda
│   └── results
└── logs                               # Log files are here
```

## Running Files
We will use a conda environment to install the required packages.
First, create a conda environment called `customer_churn_env` from the yaml file with the command:

```
conda env create --file environment.yml 
```

Second, activate the conda environment:

```
conda activate customer_churn_env
```

Finally, run the main training code:

```
python churn_library.py
```
