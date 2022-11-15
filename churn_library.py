# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import shap
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	

    logging.info(f'Reading from file {pth}')
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    logging.info('Bank data successfully loaded as dataframe')
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    # exit function with error on empty dataframe
    if df.empty:
        logging.error('Input dataframe is empty! Please input a valid dataframe')
        return

    # churn plot
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist()
    plt.savefig('./images/eda/churn_histogram.png')
    logging.info('Churn histogram saved to images directory as `churn_histogram.png`')

    # age distribution plot
    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_distribution.png')
    logging.info('Customer age distribution plot saved to images directory as `customer_age_distribution.png`')

    # marital status bar plot
    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status.png')
    logging.info('Marital status plot bar plot saved to images directory as `marital_status.png`')

    # total trans histogram
    plt.figure(figsize=(20,10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/total_trans_hist.png')
    logging.info('Total trans histogram saved to images directory as `total_trans_hist.png`')

    # features heatmap
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('./images/eda/heatmap.png')
    logging.info('Heatmap saved to images directory as `heatmap.png`')

    plt.close()



def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    logging.info('Encoding categorical columns...')

    for category in category_lst:
        cat_list = []
        cat_groups = df.groupby(category).mean()[response]

        for val in df[category]:
            cat_list.append(cat_groups.loc[val])

        df[category+response] = cat_list
        logging.info(f'Encoding {category} successfully completed!')

    return df



def perform_feature_engineering(df, response=None):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    
    df = encoder_helper(df, category_lst)
  
    y = df[response]
    X = df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass

def main():
    logging.basicConfig(
        filename='./logs/churn_logs.log',
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s [%(levelname)s] - %(funcName)s %(message)s'
    )

    bank_data = import_data(r"./data/bank_data.csv")
    perform_eda(bank_data)
    X_train, X_test, y_train, y_test = perform_feature_engineering(bank_data)


if __name__ == "__main__":
    main()