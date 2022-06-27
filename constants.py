"""
Contant holder for churn_library and tests

@author: gari.ciodaro.guerra
@date: 26-06-2022
"""
import seaborn as sns

PLOTS_STYLE = 'dark_background'

DATA_FILE = './data/bank_data.csv'

EDA_IMAGES_PATH = './images/eda/'

LOG_FILE  = './logs/churn_library.log'


LABEL_DICT= {0:'Client', 1: 'Churn'} 

EXPECTED_IMAGES_EDA_SET = {
                'churn_distribution.png', 
                'marital_status_distribution.png',
                'age_distribution.png',
                'total_trans_ct.png',
                'correlation_matrix.png'}
COLOR_PALLETTE= sns.color_palette("Reds", 5)
FIGURE_SIZE_TUPLE= (15, 7)

CATEGORICAL_FEATURES = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

CATEGORICAL_FEATURES_TARGET_ENCODED = [each+'_Churn' for each in CATEGORICAL_FEATURES]

NUMERICAL_FEATURES = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio'
]

PROCESSED_FEATURES = CATEGORICAL_FEATURES_TARGET_ENCODED + NUMERICAL_FEATURES
