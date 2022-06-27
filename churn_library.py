"""
Helper function of churn client machine learning project.

@author: gari.ciodaro.guerra
@date: 26-06-2022
"""

# import libraries
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from constants import (
    PLOTS_STYLE, 
    DATA_FILE,
    EDA_IMAGES_PATH, 
    LOG_FILE,
    EXPECTED_IMAGES_EDA_SET,
    LABEL_DICT,
    COLOR_PALLETTE,
    FIGURE_SIZE_TUPLE,
    CATEGORICAL_FEATURES,
    PROCESSED_FEATURES)

# Script configuration
mpl.style.use([PLOTS_STYLE])
#os.environ['QT_QPA_PLATFORM']='offscreen'

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
        df: (pandas.dataframe).
    """
    loaded_df = pd.read_csv(
        pth_csv, 
        delimiter= delimiter, 
        index_col= 0 if has_index_column else None)
    if is_target_required:
        loaded_df['Churn'] = loaded_df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
    return loaded_df

def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    Parameters
    ----------
        df: (pandas.dataframe)
    """
    # Setup constant
    

    # Get counts per label in churn and marital status
    df_churn = df.Churn.map(LABEL_DICT).value_counts()
    df_marital_status = df.Marital_Status.value_counts()

    # Get bar and pie plot for churn and marital status features
    for data, feature in zip([df_churn, df_marital_status], ['churn', 'marital_status']):
        fig  =  plt.figure(figsize= FIGURE_SIZE_TUPLE)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        data.plot(
            kind='bar', 
            ax= ax1,
            color=COLOR_PALLETTE)
        data.plot(
            kind='pie',
            ax= ax2, 
            autopct='%1.1f%%',
            shadow=True,
            colors=COLOR_PALLETTE)
        plt.suptitle(feature.replace('_', ' ')+ ' Distribution')
        plt.savefig(EDA_IMAGES_PATH + feature + '_distribution.png')

    # Get histogram plot of age feature.
    fig  =  plt.figure(figsize= FIGURE_SIZE_TUPLE)
    df.Customer_Age.plot(kind='hist', color= COLOR_PALLETTE[-1])
    plt.suptitle('Age Distribution.')
    plt.xlabel('Years')
    plt.savefig(EDA_IMAGES_PATH + 'age_distribution.png')

    # Get distribution plot of Total_Trans_Ct feature
    plt.figure(figsize= FIGURE_SIZE_TUPLE) 
    sns.histplot(
        df.Total_Trans_Ct, 
        stat='density', 
        kde=True,
        color= COLOR_PALLETTE[-1])
    plt.savefig(EDA_IMAGES_PATH + 'total_trans_ct.png')

    # Get pairwise correlation plot.
    plt.figure(figsize= FIGURE_SIZE_TUPLE) 
    sns.heatmap(df.corr(), annot=False, linewidths = 2)
    plt.savefig(EDA_IMAGES_PATH + 'correlation_matrix.png')



def encoder_helper(df):
    """
    Target enconding of categorical features.
    Parameters
    ----------
        df: (pandas.dataframe).
        category_list: (list) list of categorical columns.
    """
    for each_category in CATEGORICAL_FEATURES:
        # Create a dictionry that contains the mean Churn per label
        # within a given category
        holder_dictionary = df.groupby(each_category).mean()['Churn'].to_dict()
        # Target encode. Store on column with sufix '_Churn'
        # Modification happens in place. The reference dataframe change.
        df[each_category + '_Churn'] = df[each_category].map(holder_dictionary)


def perform_feature_engineering(df):
    """Select columns to learn from, split input data into training
    and testing.
    Parameters
    ----------
        df: pandas dataframe
    Returns
    ----------
        X_train: (pandas.series) train features
        X_test:  (pandas.series) train features
        y_train:  (pandas.series) train target
        y_test:  (pandas.series) train target
    """
    X = df[PROCESSED_FEATURES]
    y = df['Churn']
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