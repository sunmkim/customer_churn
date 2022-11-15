import os
import logging
import pandas as pd
import churn_library as churnlib

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
	except FileNotFoundError as err:
		logging.error("Testing import_eda ERROR: The file `bank_data.csv` wasn't found")
		raise err
	
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data ERROR: The file doesn't appear to have rows and columns")
		raise err
	logging.info("Testing import_data: SUCCESS")
	return df


def test_eda(df):
	'''
	test perform eda function
	'''
	# check that eda plot files exist
	for plot in ["churn_histogram", "customer_age_distribution", "marital_status", "total_trans_hist", "heatmap"]:
		try:
			assert os.path.exists(f"./images/eda/{plot}.png")
		except FileNotFoundError as err:
			logging.error("Testing perform_eda ERROR: One or more EDA plots are missing")
			raise err
	logging.info('Testing perform_eda: SUCCESS')


def test_encoder_helper(df):
	'''
	test encoder helper
	'''
	# check for empty df
	assert isinstance(df, pd.DataFrame)
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing encoder_helper ERROR: The dataframe doesn't appear to have rows and columns")
		raise err

	category_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
	data = churnlib.encoder_helper(df, category_cols)

	# check if categorical columns exist in df
	try:
		for col in category_cols:
			assert col in df.columns
	except AssertionError as err:
		logging.error(f"Testing encoder_helper ERROR: Missing column {col}")
		raise err
	logging.info("Testing encoder_helper: SUCCESS")

	return data


def test_feature_importance_plot():
	'''
	test feature_importance_plot
	'''
	# check that feature importance plot exist
	try:
		assert os.path.exists('./images/results/feature_importance.png') 
	except AssertionError as err:
		logging.error(f"Testing feature_importance_plot ERROR: feature_importance.png plot not found.")
		raise err
	logging.info("Testing feature_importance_plot: SUCCESS")


def test_classification_report_image():
	# check that featture importance plot exist
	try:
		assert os.path.exists('./images/results/logistic_results.png')
		assert os.path.exists('./images/results/randomforest_results.png') 
	except AssertionError as err:
		logging.error(f"Testing classification_report_image ERROR: results plot not found.")
		raise err
	logging.info("Testing classification_report_image: SUCCESS")


def test_save_roc_plots():
	'''
	test save_roc_plots
	'''
	# check that feature importance plot exist
	try:
		assert os.path.exists('./images/results/roc_curve.png') 
	except AssertionError as err:
		logging.error(f"Testing save_roc_plots ERROR: roc_curve.png plot not found.")
		raise err
	logging.info("Testing save_roc_plots: SUCCESS")


def test_feature_importance_plot():
	# check that featture importance plot exist
	try:
		assert os.path.exists('./images/results/feature_importance.png') 
	except AssertionError as err:
		logging.error(f"Testing feature_importance_plot ERROR: feature_importance.png plot not found.")
		raise err
	logging.info("Testing feature_importance_plot: SUCCESS")


def test_perform_feature_engineering(encoded_df):
	'''
	test perform_feature_engineering
	'''
	X_train, X_test, y_train, y_test = churnlib.perform_feature_engineering(encoded_df)

	try:
		# check that data are dataframes
		assert isinstance(X_train, pd.DataFrame)
		assert isinstance(X_test, pd.DataFrame)
		assert isinstance(y_train, pd.Series)
		assert isinstance(y_test, pd.Series)

		# check that dataframes are not empty
		assert X_train.shape[0] > 0
		assert X_test.shape[0] > 0
		assert y_train.shape[0] > 0
		assert y_test.shape[0] > 0

		# check that dataframes match for train and test X and y
		assert len(X_train) == len(y_train)
		assert len(X_test) == len(y_test)
	except AssertionError as err:
		logging.error(f"Testing perform_feature_engineering ERROR: Error with dataframe type or shape")
		raise err
	logging.info("Testing perform_feature_engineering: SUCCESS")

	return X_train, X_test, y_train, y_test


def test_train_models(X_train, X_test, y_train, y_test):
	'''
	test train_models
	'''

	# train model
	churnlib.train_models(X_train, X_test, y_train, y_test)

	# check model pickle files exist
	for model in ['./models/logistic_model.pkl', './models/rfc_model.pkl']:
		try:
			assert os.path.exists(model) 
		except AssertionError as err:
			logging.error(f"Testing train_models ERROR: Model {model} not found")
			raise err
	logging.info("Testing train_models: SUCCESS")


if __name__ == "__main__":
	logging.info('Begin testing churn_library...')
	
	test_df = test_import()
	test_eda(test_df)
	encoded_df= test_encoder_helper(test_df)
	X_train, X_test, y_train, y_test = test_perform_feature_engineering(encoded_df)
	test_train_models(X_train, X_test, y_train, y_test)
	
	# test that plots/images are generated
	test_classification_report_image()
	test_save_roc_plots()
	test_feature_importance_plot()

	logging.info('Testing completed!')
