"""
Tesing suit for Helper function of churn client machine learning project.

@author: gari.ciodaro.guerra
@date: 26-06-2022
"""

import os
import glob
import logging
import pytest
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering
#import churn_library_solution as cls

# Empty folders for testing.
files = glob.glob('./images/eda/*')
for file in files:
    os.remove(file)

# Configuring logs
logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

TESTING_PATHS_CSV  = ["./data/bank_data.csv"]

EXPECTED_FILES_SET = {
                'churn_distribution.png', 
                'marital_status_distribution.png',
                'age_distribution.png',
                'total_trans_ct.png',
                'correlation_matrix.png'}

@pytest.fixture(
    scope="module",
    params=TESTING_PATHS_CSV)
def path(request):
  value = request.param
  yield  value

def test_import_data(path):
    """Test for loading correclty input data"""
    try:
        df = import_data(
            pth_csv= path, 
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

def test_perform_eda():
    try:
        perform_eda(pytest.test_data_df)
        actual_files_set = set(os.listdir('./images/eda'))
        difference_set = EXPECTED_FILES_SET - actual_files_set
        assert len(difference_set)==0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(f"Testing perform_eda: {difference_set} plots not found")
        raise err




if __name__ == "__main__":
    df = test_import_data(TESTING_PATHS_CSV[0])
    test_perform_eda(df)
    #test_import_data(TESTING_PATHS_CSV[1])
    #test_import_data("./data/bx.csv")