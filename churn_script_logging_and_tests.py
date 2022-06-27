"""
Tesing suit for Helper function of churn client machine learning project.

@author: gari.ciodaro.guerra
@date: 26-06-2022
"""

import os
import glob
import logging
import pytest
from churn_library import (
    import_data, 
    perform_eda, 
    encoder_helper, 
    perform_feature_engineering )
from constants import (
    DATA_FILE,
    EDA_IMAGES_PATH, 
    LOG_FILE,
    EXPECTED_IMAGES_EDA_SET,
    CATEGORICAL_FEATURES,
    PROCESSED_FEATURES,
    CATEGORICAL_FEATURES_TARGET_ENCODED)
#import churn_library_solution as cls

# Empty folders for testing.
files = glob.glob(EDA_IMAGES_PATH+'*')
for file in files:
    os.remove(file)

# Configuring logs
logging.basicConfig(
    filename= LOG_FILE,
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import_data():
    """Test for loading correclty input data"""
    try:
        df = import_data(
            pth_csv= DATA_FILE, 
            delimiter=',', 
            has_index_column= True, 
            is_target_required= True)
        logging.info("Testing import_data: SUCCESS")
        pytest.test_data_df = df
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        return df
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_perform_eda(df):
    try:
        perform_eda(df)
        actual_files_set = set(os.listdir('./images/eda'))
        difference_set = EXPECTED_IMAGES_EDA_SET - actual_files_set
        assert len(difference_set)==0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(f"Testing perform_eda: {' '.join(difference_set)} plots not found")
        raise err


def test_encoder_helper(df):
    try:
        encoder_helper(df)
        # Call target encoded features.
        df[CATEGORICAL_FEATURES_TARGET_ENCODED]
        logging.info('Testing encoder_helper: SUCCESS')

    except KeyError as err:
        logging.error(f"Testing encoder_helper: {err}")
        raise err

def test_perform_feature_engineering(df):
    try:
        expected_rows_int, expected_columns_int  = df.shape[0], len(PROCESSED_FEATURES) 
        output_df_tuple = perform_feature_engineering(df)
        X_train, X_test, y_train, y_test = output_df_tuple
        assert len(output_df_tuple) == 4, 'Output of function must be a 4 element tuple'

        logging.info(f"""
        Testing perform_feature_engineering: 
        input df shape {df.shape}
        Expected number of features {len(PROCESSED_FEATURES)}
        X_train shape {X_train.shape}
        X_test shape {X_test.shape}
        y_train shape {y_train.shape}
        y_test shape {y_test.shape}""")

        columns_equality = (
            X_train.shape[1] == X_test.shape[1]  == expected_columns_int 
            and 
            len(y_train.shape) == len(y_test.shape) == 1 )

        row_equality = (
            X_train.shape[0] + X_test.shape[0] ==
            y_train.shape[0] + y_test.shape[0] == expected_rows_int )
        assert columns_equality, 'dimension incosistency in columns'
        assert row_equality, 'dimension incosistency in rows'
        
        logging.info('Testing perform_feature_engineering: SUCCESS')

    except AssertionError as err:
        logging.error(f'Testing perform_feature_engineering: {err}')
        raise err

if __name__ == "__main__":
    df = test_import_data()
    test_perform_eda(df)
    test_encoder_helper(df)
    test_perform_feature_engineering(df)
    #test_import_data(TESTING_PATHS_CSV[1])
    #test_import_data("./data/bx.csv")