"""
Tesing suit for Helper function of churn client machine learning project.

@author: gari.ciodaro.guerra
@date: 26-06-2022
"""

import os
import glob
import logging
import joblib

from churn_library import (
    import_data,
    perform_eda,
    encoder_helper,
    perform_feature_engineering,
    train_models,
    feature_importance_plot,
)
from constants import (
    DATA_FILE,
    EDA_IMAGES_PATH,
    LOG_FILE,
    EXPECTED_IMAGES_EDA_SET,
    PROCESSED_FEATURES,
    CATEGORICAL_FEATURES_TARGET_ENCODED,
    MODEL_STORE_PATH,
    EXPECTED_MODELS_SET,
    RESULTS_IMAGES_PATH,
    EXPECTED_IMAGES_RESULST_SET,
    NAME_FEATURE_IMPORTANCE_PLOT,
)

# import churn_library_solution as cls

# Empty folders for testing.
for paths in [EDA_IMAGES_PATH, MODEL_STORE_PATH, RESULTS_IMAGES_PATH]:
    files = glob.glob(paths + "*")
    for file in files:
        os.remove(file)

# Configuring logs
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import_data():
    """Test for loading correclty input data"""
    try:
        data = import_data(
            pth_csv=DATA_FILE,
            delimiter=",",
            has_index_column=True,
            is_target_required=True,
        )
        logging.info("Testing import_data: SUCCESS")
        assert data.shape[0] > 0
        assert data.shape[1] > 0
        return data
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_perform_eda(data):
    """Test EDA analysis. Check if images were created on images/eda.
    Location of images on EDA_IMAGES_PATH on constants.py
    Parameters
    ----------
        data: (pandas.dataframe)
    """
    try:
        perform_eda(data)
        actual_files_set = set(os.listdir(EDA_IMAGES_PATH))
        difference_set = EXPECTED_IMAGES_EDA_SET - actual_files_set
        assert len(difference_set) == 0
        logging.info("Testing perform_eda: SUCCESS")
        logging.info(
            f"""
        Testing perform_eda: Images created in {EDA_IMAGES_PATH}.
        {', '.join(EXPECTED_IMAGES_EDA_SET)}"""
        )
    except AssertionError as err:
        logging.error(
            f"Testing perform_eda: {', '.join(difference_set)} plots not found"
        )
        raise err


def test_encoder_helper(data):
    """Test encoding function. Modifies input dataframe in place.
    CATEGORICAL_FEATURES_TARGET_ENCODED is a list of features to target
    enconde.
    Parameters
    ----------
        data: (pandas.dataframe)
    """
    try:
        encoder_helper(data)
        # Call target encoded features.
        data[CATEGORICAL_FEATURES_TARGET_ENCODED]
        logging.info(
            f"""
        Testing encoder_helper: SUCCESS.
        features encoded. {', '.join(CATEGORICAL_FEATURES_TARGET_ENCODED)}
        """
        )
    except KeyError as err:
        logging.error(f"Testing encoder_helper: {err}")
        raise err


def test_perform_feature_engineering(data):
    """Test encoding function.

    Parameters
    ----------
        data: (pandas.dataframe)
    Returns
    -------
        features_train: (pandas.series) features training.
        features_test: (pandas.series) features testing.
        target_train: (pandas.series) target training.
        target_test: (pandas.series) target testing.
    """
    try:
        expected_rows_int, expected_columns_int = data.shape[0], len(
            PROCESSED_FEATURES)
        output_df_tuple = perform_feature_engineering(data)
        features_train, features_test, target_train, target_test = output_df_tuple
        assert len(
            output_df_tuple) == 4, "Output of function must be a 4 element tuple"

        logging.info(
            f"""
        Testing perform_feature_engineering:
        input data shape {data.shape}
        Expected number of features {len(PROCESSED_FEATURES)}
        features_train shape {features_train.shape}
        features_test shape {features_test.shape}
        target_train shape {target_train.shape}
        target_test shape {target_test.shape}"""
        )

        columns_equality = (
            features_train.shape[1] == features_test.shape[1] == expected_columns_int
            and len(target_train.shape) == len(target_test.shape) == 1
        )

        row_equality = (
            features_train.shape[0] + features_test.shape[0]
            == target_train.shape[0] + target_test.shape[0]
            == expected_rows_int
        )
        assert columns_equality, "dimension incosistency in columns"
        assert row_equality, "dimension incosistency in rows"

        logging.info("Testing perform_feature_engineering: SUCCESS")
        return features_train, features_test, target_train, target_test

    except AssertionError as err:
        logging.error(f"Testing perform_feature_engineering: {err}")
        raise err


def test_train_models(
        features_train,
        features_test,
        target_train,
        target_test):
    """Checks RandomForestClassifier and LogisticRegression
    models are stored after execution train_models function.

    Parameters
    ----------
        features_train: (pandas.series) features training.
        features_test: (pandas.series) features testing.
        target_train: (pandas.series) target training.
        target_test: (pandas.series) target testing.
    """
    try:
        train_models(features_train, features_test, target_train, target_test)

        actual_files_set = set(
            os.listdir(MODEL_STORE_PATH) + os.listdir(RESULTS_IMAGES_PATH)
        )
        difference_set = (EXPECTED_MODELS_SET.union(
            EXPECTED_IMAGES_RESULST_SET) - actual_files_set)

        assert len(difference_set) == 0
        logging.info(
            f"""
        Testing test_train_models:
        models were created in {MODEL_STORE_PATH}.
        {', '.join(EXPECTED_MODELS_SET)}"""
        )

        logging.info(
            f"""
        Testing test_train_models:
        images results were created in {RESULTS_IMAGES_PATH}.
        {', '.join(EXPECTED_IMAGES_RESULST_SET)}"""
        )

        logging.info("Testing train_models: SUCCESS")

    except AssertionError as err:
        logging.error(
            f"""
        Testing train_models: {', '.join(difference_set)} not found
        """
        )
        raise err


def test_feature_importance_plot(data, model_tree, path_to_store):
    """
    checks that feature_importance_plot stores the feature importance
    plots in path_to_store.

    Parameters
    ----------

    """
    try:
        # delete old feature_importance plot with the same name.
        if NAME_FEATURE_IMPORTANCE_PLOT in os.listdir(path_to_store):
            os.remove(path_to_store + NAME_FEATURE_IMPORTANCE_PLOT)
        feature_importance_plot(data, model_tree, path_to_store)
        # the the file was created.
        assert NAME_FEATURE_IMPORTANCE_PLOT in os.listdir(path_to_store)
        logging.info("Testing feature_importance_plot: SUCCESS")

    except AssertionError as err:
        logging.error(
            f"""
        Testing feature_importance_plot:
        feature importance {path_to_store+ NAME_FEATURE_IMPORTANCE_PLOT}
        was not found.
        """
        )
        raise err


if __name__ == "__main__":
    logging.info("Testing import_data: START")
    data_for_test = test_import_data()
    logging.info("Testing import_data: END")
    logging.info("Testing perform_eda: START")
    test_perform_eda(data_for_test)
    logging.info("Testing perform_eda: END")
    logging.info("Testing encoder_helper: START")
    test_encoder_helper(data_for_test)
    logging.info("Testing encoder_helper: END")
    logging.info("Testing perform_feature_engineering: START")
    features_train, features_test, target_train, target_test = test_perform_feature_engineering(
        data_for_test)
    logging.info("Testing perform_feature_engineering: END")
    logging.info("Testing train_models: START")
    test_train_models(features_train, features_test, target_train, target_test)
    logging.info("Testing train_models: END")
    logging.info("Testing feature_importance_plot: START")
    test_feature_importance_plot(
        features_test,
        joblib.load(MODEL_STORE_PATH + "RandomForest.pkl"),
        path_to_store="./images/feature_importance/",
    )
    logging.info("Testing feature_importance_plot: END")
