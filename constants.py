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
