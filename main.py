#!/usr/bin/env python
# coding: utf-8



"""Library & data imports"""
import os
import logging

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

from joblib import dump, load





logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


LOGGER.info("Read in and prepare csv data")
DATA = pd.read_csv(r"~/Desktop/DataCoSupplyChainDataset.csv", encoding='cp1252')
SUB_DATA = DATA[DATA['Customer Country'] == 'EE. UU.']


LOGGER.info(r"Downsample noncancelled data to ~7\% industry average")

def classify_shipping(del_input) -> int:
    """Function to classify an order status as cancelled or not.

    Arguments:
        - delivery status (string).

    Returns:
        - Flag for cancelled/non-cancelled (int).
    """
    if del_input == 'Shipping canceled':
        return 1

    return 0

BALANCE = SUB_DATA.loc[:, 'Delivery Status'].apply(lambda x: classify_shipping(x))
BALANCE = pd.DataFrame(BALANCE)

DF_MAJORITY = BALANCE[BALANCE['Delivery Status'] == 0]
DF_MINORITY = BALANCE[BALANCE['Delivery Status'] == 1]


NEW_MAJORITY_NUMBER = ((DF_MINORITY.shape[0]/0.075) - DF_MINORITY.shape[0])
NEW_MAJORITY_NUMBER = int(round(NEW_MAJORITY_NUMBER))


DF_MAJORITY_DOWNSAMPLED = resample(SUB_DATA[SUB_DATA['Delivery Status'] != 'Shipping canceled'],
                                   replace=False, n_samples=NEW_MAJORITY_NUMBER, random_state=29)

DF_MINORITY = SUB_DATA[SUB_DATA['Delivery Status'].apply(lambda x: classify_shipping(x)) == 1]

DF_DOWNSAMPLED = pd.concat([DF_MAJORITY_DOWNSAMPLED, DF_MINORITY], axis=0)


LOGGER.info("Create categorical variables to be used in this analysis")


ELECTRONICS = ['Electronics', 'Music', 'DVDs', 'Video Games', 'CDs ', 'Consumer Electronics',
               'Cameras ', 'Computers']
APPAREL = ["Girls' Apparel", "Women's Apparel", "Women's Clothing", "Men's Footwear",
           "Men's Clothing", "Children's Clothing", 'Baby ', 'Health and Beauty']
SPORTS = ['Sporting Goods', 'Cardio Equipment', 'Cleats', 'Shop By Sport', 'Hunting & Shooting',
          'Tennis & Racquet', 'Baseball & Softball', 'Fitness Accessories', 'Golf Balls',
          'Lacrosse', 'Boxing & MMA', 'Soccer', 'Fishing', 'Camping & Hiking', 'Hockey',
          'Basketball', 'Strength Training', 'Golf Gloves', 'Golf Bags & Carts', 'Golf Shoes',
          'Golf Apparel', "Women's Golf Clubs", "Men's Golf Clubs", 'Water Sports',
          'Indoor/Outdoor Games', "Kids' Golf Clubs", 'Toys', 'As Seen on  TV!',
          'Accessories', 'Trade-In']

def cat_buckets(product) -> str:
    """Function to categorize the string inputs. This simplification to 3 major levels makes
       this attribute much
       easier to interpret when used in the final model.

    Arguments:
        - String 'Category Name'

    Returns:
        - String categorization into 1 of 3 major buckets: Electronics, Apparel, Sports
    """
    if product in ELECTRONICS:
        return 'Electronics'
    if product in APPAREL:
        return 'Apparel'
    if product in SPORTS:
        return 'Sports'

    return 'Other'

DF_DOWNSAMPLED['Category Buckets'] = DF_DOWNSAMPLED['Category Name'].apply(lambda x: cat_buckets(x))


"""Create time attributes"""
CURRENT_DATE_FORMAT = pd.to_datetime(DF_DOWNSAMPLED['order date (DateOrders)'],
                                     format='%m/%d/%Y %H:%M')

DF_DOWNSAMPLED['week-date'] = CURRENT_DATE_FORMAT.apply(lambda x: x.strftime('%u'))



def classify_region(state) -> str:
    """Classify  a stat's region
        Argument: state (string)
        Returns: region (string)
    """
    if state in ('CA', 'OR', 'WA', 'AK', 'HI'):
        return 'pacific'
    if state in ('AZ', 'NM', 'CO', 'UT', 'NV', 'ID', 'WY', 'MT'):
        return 'mountain'
    if state in ('ND', 'MN', 'SD', 'IA', 'NE', 'MO', 'KS'):
        return 'west north central'
    if state in ('WI', 'MI', 'IL', 'IN', 'OH'):
        return 'east north central'
    if state in ('OK', 'TX', 'AR', 'LA'):
        return 'west south central'
    if state in ('KY', 'TN', 'MS', 'AL'):
        return 'east south central'
    if state in ('WV', 'MD', 'DE', 'VA', 'NC', 'SC', 'GA', 'FL', 'DC'):
        return 'south atlantic'
    if state in ('NY', 'PA', 'NJ'):
        return 'middle atlantic'
    if state in ('ME', 'VT', 'NH', 'MA', 'CT', 'RI'):
        return 'new england'
    return None

DF_DOWNSAMPLED['customer-region'] = DF_DOWNSAMPLED['Customer State'].apply(lambda x:
                                                                           classify_region(x))


def cancelled(order) -> int:
    """Function to classify whether an order was cancelled or not.

        Arguments:
            -Order status
        Returns:
            -One-hot encoding flag: 1, means order was cancelled; otherwise, 0.
    """
    if order == 'Shipping canceled':
        return 1
    return 0

DF_DOWNSAMPLED['cancelled'] = DF_DOWNSAMPLED['Delivery Status'].apply(lambda k: cancelled(k))


"""Let's focus on order attributes important to us."""
SUB_DATA_FEATURES = DF_DOWNSAMPLED.loc[:, ['Customer Segment', 'Order Item Quantity',
                                           'Category Buckets', 'week-date', 'customer-region',
                                           'cancelled']]




LOGGER.info("Split working data to training and testing sets")

X = SUB_DATA_FEATURES.iloc[:, [0, 1, 2, 3, 4]]
Y = SUB_DATA_FEATURES.iloc[:, [-1]]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.25)




"""Get training data & estabilsh base levels"""
X_TRAIN = pd.get_dummies(X_TRAIN, columns=['Customer Segment', 'Order Item Quantity',
                                           'Category Buckets', 'week-date', 'customer-region'])
# Y_TRAIN

TRAIN_FEATURES_X = X_TRAIN.loc[:, ['Customer Segment_Corporate', 'Customer Segment_Home Office',
                                   'Order Item Quantity_2', 'Order Item Quantity_3',
                                   'Order Item Quantity_4', 'Order Item Quantity_5',
                                   'Category Buckets_Electronics', 'Category Buckets_Sports',
                                   'week-date_2', 'week-date_3', 'week-date_4', 'week-date_5',
                                   'week-date_6', 'week-date_7',
                                   'customer-region_east north central',
                                   'customer-region_east south central',
                                   'customer-region_middle atlantic', 'customer-region_mountain',
                                   'customer-region_new england', 'customer-region_south atlantic',
                                   'customer-region_west north central',
                                   'customer-region_west south central']]
TRAIN_Y = Y_TRAIN


"""Get test data & establish test levels"""
X_TEST = pd.get_dummies(X_TEST, columns=['Customer Segment', 'Order Item Quantity',
                                         'Category Buckets', 'week-date', 'customer-region'])
# Y_TEST = y_test

TEST_FEATURES_X = X_TEST.loc[:, ['Customer Segment_Corporate', 'Customer Segment_Home Office',
                                 'Order Item Quantity_2', 'Order Item Quantity_3',
                                 'Order Item Quantity_4', 'Order Item Quantity_5',
                                 'Category Buckets_Electronics', 'Category Buckets_Sports',
                                 'week-date_2', 'week-date_3', 'week-date_4', 'week-date_5',
                                 'week-date_6', 'week-date_7',
                                 'customer-region_east north central',
                                 'customer-region_east south central',
                                 'customer-region_middle atlantic',
                                 'customer-region_mountain', 'customer-region_new england',
                                 'customer-region_south atlantic',
                                 'customer-region_west north central',
                                 'customer-region_west south central']]
TEST_Y = Y_TEST




LOGGER.info("Load saved model; or create first model (Model 1)")

FILENAME = 'lr_model.sav'

if os.path.isfile('lr_model.sav'):
    CURRENT_MODEL = load(FILENAME)
else:
    CURRENT_MODEL = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.75,
                                       class_weight='balanced', C=0.5, verbose=False)


LOGGER.info("Fit model (Model 1)")

CURRENT_MODEL.fit(TRAIN_FEATURES_X, TRAIN_Y)




Y_PRED_TRAIN_PROBS = pd.DataFrame(CURRENT_MODEL.predict_proba(TEST_FEATURES_X))
MODEL1_SCORE = roc_auc_score(y_true=TEST_Y, y_score=Y_PRED_TRAIN_PROBS.iloc[:, 1],
                             multi_class='ovo')





LOGGER.info("Create new model from this data (Model 2)")

LR_ALT = LogisticRegression()

PARAMETERS = {
    'C': [0.2, .33, 0.5, 0.66, 0.75, 1, 1.5, 5, 10, 20],
    'class_weight': ['balanced'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'l1_ratio': [0.2, .33, 0.5, 0.66, 0.75]}

LOGGER.info("Fine tune model and fit it (Model 2)")

NEW_MODEL = RandomizedSearchCV(LR_ALT, PARAMETERS, cv=4, n_iter=15)


# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore")
NEW_MODEL.fit(TRAIN_FEATURES_X, TRAIN_Y)



Y_PRED_TRAIN_PROBS = pd.DataFrame(NEW_MODEL.predict_proba(TEST_FEATURES_X))
MODEL2_SCORE = roc_auc_score(y_true=TEST_Y, y_score=Y_PRED_TRAIN_PROBS.iloc[:, 1],
                             multi_class='ovo')




"""
    The following scenarios outlined.
    Do NOT update model when:
        New model ROC_AIC score goes down
        New model ROC_AIC score is the same as old model
        New model ROC_AIC score increases only slightly (by 0.01)
        New model ROC_AIC score increases astoundingly (0.3)
        New model ROC_AIC score is greater than 0.9
    Do Update model when:
        None of the above conditions are meet
"""

LOGGER.info("Determine whether existing model needs to be updated")
MODEL_SCORE_DIFF = (MODEL2_SCORE - MODEL1_SCORE)

if MODEL_SCORE_DIFF <= 0:
    dump(CURRENT_MODEL, FILENAME)
elif MODEL_SCORE_DIFF < 0.01:
    dump(CURRENT_MODEL, FILENAME)
elif MODEL_SCORE_DIFF > 0.3:
    dump(CURRENT_MODEL, FILENAME)
elif MODEL2_SCORE >= 0.9:
    dump(CURRENT_MODEL, FILENAME)
else:
    dump(NEW_MODEL, FILENAME)

LOGGER.info("Done")
