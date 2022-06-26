"""
Pytest Namespace configuration. Allow for sharing pandas.dataframe 
transformations.

@author: gari.ciodaro.guerra
@date: 26-06-2022
"""

import pytest

def df_plugin():
    return None

# Creating a Dataframe object 'pytest.test_data_df' in Namespace
def pytest_configure():
  pytest.test_data_df = df_plugin()