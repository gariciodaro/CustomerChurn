"""
Helper function of churn client machine learning project.

@author: gari.ciodaro.guerra
@date: 26-06-2022
"""

# import libraries
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay, classification_report
import shap
import joblib


from constants import (
    PLOTS_STYLE,
    EDA_IMAGES_PATH,
    LABEL_DICT,
    COLOR_PALLETTE,
    FIGURE_SIZE_TUPLE,
    CATEGORICAL_FEATURES,
    PROCESSED_FEATURES,
    MODEL_STORE_PATH,
    RESULTS_IMAGES_PATH,
    NAME_FEATURE_IMPORTANCE_PLOT,
    DATA_FILE
)


# Script configuration
mpl.style.use([PLOTS_STYLE])
# os.environ['QT_QPA_PLATFORM']='offscreen'


def import_data(pth_csv, delimiter, has_index_column, is_target_required):
    """
    returns dataframe for the csv found at pth

    Parameters
    ----------
        pth_csv: (str) a path to the csv
        delimiter: (str) valid delimiter for csv.
        has_index_column: (boolean).
        is_target_required: (boolean). If true, Attrition_Flag
        feature will be used to compute target as Churn.

    Returns
    ----------
        data_frame: (pandas.dataframe).
    """
    loaded_df = pd.read_csv(
        pth_csv, delimiter=delimiter, index_col=0 if has_index_column else None
    )
    if is_target_required:
        loaded_df["Churn"] = loaded_df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
    return loaded_df


def perform_eda(data_frame):
    """
    perform eda on data_frame and save figures to images folder

    Parameters
    ----------
        data_frame: (pandas.dataframe)
    """
    # Get counts per label in churn and marital status
    df_churn = data_frame.Churn.map(LABEL_DICT).value_counts()
    df_marital_status = data_frame.Marital_Status.value_counts()

    # Get bar and pie plot for churn and marital status features
    for data, feature in zip(
        [df_churn, df_marital_status], ["churn", "marital_status"]
    ):
        fig = plt.figure(figsize=FIGURE_SIZE_TUPLE)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        data.plot(kind="bar", ax=ax1, color=COLOR_PALLETTE)
        data.plot(
            kind="pie", ax=ax2, autopct="%1.1f%%", shadow=True, colors=COLOR_PALLETTE
        )
        plt.suptitle(feature.replace("_", " ") + " Distribution")
        plt.savefig(EDA_IMAGES_PATH + feature + "_distribution.png")
        plt.close()

    # Get histogram plot of age feature.
    fig = plt.figure(figsize=FIGURE_SIZE_TUPLE)
    data_frame.Customer_Age.plot(kind="hist", color=COLOR_PALLETTE[-1])
    plt.suptitle("Age Distribution.")
    plt.xlabel("Years")
    plt.savefig(EDA_IMAGES_PATH + "age_distribution.png")
    plt.close()

    # Get distribution plot of Total_Trans_Ct feature
    plt.figure(figsize=FIGURE_SIZE_TUPLE)
    sns.histplot(
        data_frame.Total_Trans_Ct, stat="density", kde=True, color=COLOR_PALLETTE[-1]
    )
    plt.savefig(EDA_IMAGES_PATH + "total_trans_ct.png")
    plt.close()

    # Get pairwise correlation plot.
    plt.figure(figsize= (15, 15) )
    sns.heatmap(data_frame.corr(), annot=False, linewidths=2)
    plt.savefig(EDA_IMAGES_PATH + "correlation_matrix.png")
    plt.close()


def encoder_helper(data_frame):
    """
    Target enconding of categorical features. Transformation done in place.

    Parameters
    ----------
        data_frame: (pandas.dataframe).
        category_list: (list) list of categorical columns.
    """
    for each_category in CATEGORICAL_FEATURES:
        # Create a dictionry that contains the mean Churn per label
        # within a given category
        holder_dictionary = data_frame.groupby(each_category).mean()["Churn"].to_dict()
        # Target encode. Store on column with sufix '_Churn'
        # Modification happens in place. The reference dataframe change.
        data_frame[each_category + "_Churn"] = data_frame[each_category].map(
            holder_dictionary
        )


def perform_feature_engineering(data_frame):
    """Select columns to learn from, split input data into training
    and testing.

    Parameters
    ----------
        data_frame: pandas dataframe
    Returns
    ----------
    features_train: (pandas.series) train features
    features_test:  (pandas.series) train features
    target_train:  (pandas.series) train target
    target_test:  (pandas.series) train target
    """
    # Select only relevant features
    features = data_frame[PROCESSED_FEATURES]
    target = data_frame["Churn"]
    # make the split.
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )
    return features_train, features_test, target_train, target_test


def train_models(features_train, features_test, target_train, target_test):
    """
    train and store RandomForestClassifier and LogisticRegression models.
    RandomForestClassifier is optimized wirh GridSearch and Crossvalidation.
    LogisticRegression serves as baseline. Store roc curves.

    Parameters
    ----------
        features_train: (pandas.series) features training.
        features_test: (pandas.series) features test.
        target_train: (pandas.series) target training.
        target_test: (pandas.series) test training.
    """
    random_forest_base = RandomForestClassifier(random_state=42)
    logistic_regression = LogisticRegression(solver="lbfgs", max_iter=100)
    parameter_grid_dictionary = {
        "n_estimators": [200, 500],
        #        'max_features': ['auto', 'sqrt'],
        "max_depth": [4, 5, 100],
        #        'criterion' :['gini', 'entropy']
    }
    grid_search_random_forest = GridSearchCV(
        estimator=random_forest_base, param_grid=parameter_grid_dictionary, cv=2
    )
    grid_search_random_forest.fit(features_train, target_train)

    random_forest = grid_search_random_forest.best_estimator_
    logistic_regression.fit(features_train, target_train)

    # store models as pickle files.
    joblib.dump(random_forest, MODEL_STORE_PATH + "RandomForest.pkl")

    joblib.dump(logistic_regression, MODEL_STORE_PATH + "LogisticRegression.pkl")

    # plot classification report and ROC curves for both models.
    for model_name in ["Random Forest", "Logistic Regression"]:
        model_name_prepared = model_name.lower().replace(" ", "_")
        fig = plt.figure(figsize=FIGURE_SIZE_TUPLE)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        font_dictionary = {"fontsize": 10, "fontproperties": "monospace"}
        ax1.text(0.0, 1.0, model_name + " Train", fontdict=font_dictionary)
        ax1.text(
            0.01,
            0.7,
            str(
                classification_report(
                    target_train,
                    eval(model_name_prepared + ".predict(features_train)"),
                )
            ),
            fontdict=font_dictionary,
        )
        ax1.text(0.0, 0.6, model_name + " Test", fontdict=font_dictionary)
        ax1.text(
            0.01,
            0.3,
            str(
                classification_report(
                    target_test,
                    eval(model_name_prepared + ".predict(features_test)"),
                )
            ),
            fontdict=font_dictionary,
        )
        ax1.axis("off")
        RocCurveDisplay.from_estimator(
            eval(model_name_prepared), features_test, target_test, ax=ax2
        )
        plt.savefig(
            RESULTS_IMAGES_PATH + f"classification_report_{model_name_prepared}.png"
        )
        plt.close()


def feature_importance_plot(data_frame, model_tree, path_to_store):
    """
    Used game theory from shap value to determine feature importance of
    input data.

    Parameters
    ----------
        data_frame: (pandas.series) features.
        path_to_store: (str) where to store feature importance image
        model_tree: tree based estimator.
    """
    explainer = shap.TreeExplainer(model_tree)
    shap_values = explainer.shap_values(data_frame)
    shap.summary_plot(
        shap_values,
        data_frame,
        plot_type="bar",
        plot_size= (25, 7),
        show=False,
        class_names=LABEL_DICT,
        axis_color="#FFFFFF",
    )
    plt.savefig(path_to_store + NAME_FEATURE_IMPORTANCE_PLOT)
    plt.close()


if __name__ == "__main__":
    # read data. User can provide location. 
    # all plots will be overwritten.
    path_to_data = sys.argv[1]
    data_for_model = import_data(
        pth_csv= path_to_data,
        delimiter=",",
        has_index_column=True,
        is_target_required=True,
    )
    # perform eda. store eda plots
    perform_eda(data_for_model)
    # apply target encode
    encoder_helper(data_for_model)
    # select features for modeling. Split into train and test.
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = perform_feature_engineering(data_for_model)
    # perfom the training. Stores models on file system.
    #train_models(X_train, X_test, y_train, y_test)
    # compute and store shap values (features importance) on given data.
    feature_importance_plot(
        data_frame=X_test,
        model_tree=joblib.load(MODEL_STORE_PATH + "RandomForest.pkl"),
        path_to_store="./images/feature_importance/",
    )
