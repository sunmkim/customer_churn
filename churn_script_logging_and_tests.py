import os
import logging
import pandas as pd
import pytest
import churn_library as churnlib
from pathlib import Path

logging.basicConfig(
    filename='./logs/churn_test.log',
    level = logging.INFO,
    filemode='w',
    format='%(asctime)s [%(levelname)s] - %(funcName)s - %(message)s'
)

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = churnlib.import_data("./data/bank_data.csv")
		assert isinstance(df, pd.DataFrame)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file `bank_data.csv` wasn't found")
		raise err
	
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err
	return df


def test_eda(test_import):
	'''
	test perform eda function
	'''
	plot_types = ["churn_histogram", "customer_age_distribution", "marital_status", "total_trans_hist", "heatmap"]
	for plot in plot_types:
		try:
				with open("./images/eda/%s.png" % plot, 'r'):
					logging.info("Testing perform_eda on %s.png: SUCCESS" % plot)
		except FileNotFoundError as err:
			logging.error("Testing perform_eda: generated images missing")
			raise err


def test_encoder_helper(df):
	'''
	test encoder helper
	'''
	# Check for empty df
	assert isinstance(df, pd.DataFrame)
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("ERROR: The dataframe doesn't appear to have rows and columns")
		raise err

	category_cols = [
		'Gender',
		'Education_Level',
		'Marital_Status',
		'Income_Category',
		'Card_Category'
	]

	data = churnlib.encoder_helper(df, category_cols)

	# Check if categorical columns exist in df
	try:
			for col in category_cols:
				assert col in df.columns
	except AssertionError as err:
			logging.error("ERROR: Missing a category column")
			raise err
	logging.info("SUCCESS: Categorical columns correctly encoded.")

	return data



def test_perform_feature_engineering(encoded_df):
	'''
	test perform_feature_engineering
	'''
	X_train, X_test, y_train, y_test = churnlib.perform_feature_engineering(encoded_df)

	try:
			assert len(X_train) == len(y_train)
			assert len(X_test) == len(y_test)
	except AssertionError as err:
			logging.error("ERROR: The shape of train test splits don't match")
			raise err
	logging.info("SUCCESS: Train test successfully splitted.")

	return X_train, X_test, y_train, y_test


def test_train_models(X_train, X_test, y_train, y_test):
	'''
	test train_models
	'''

	# Train model
	churnlib.train_models(X_train, X_test, y_train, y_test)

	# Check if model were saved after done training
	path = Path("./models")

	models = ['logistic_model.pkl', 'rfc_model.pkl']

	for model_name in models:
		model_path = path.joinpath(model_name)
		try:
			assert model_path.is_file()
		except AssertionError as err:
			logging.error("ERROR: Models not found.")
			raise err
	logging.info("SUCCESS: Models successfully saved!")



if __name__ == "__main__":
	test_df = test_import()
	test_eda(test_df)
	encoded_df= test_encoder_helper(test_df)
	X_train, X_test, y_train, y_test = test_perform_feature_engineering(encoded_df)
	test_train_models(X_train, X_test, y_train, y_test)
