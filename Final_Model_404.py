#!/usr/bin/env python
# coding: utf-8



"""Library & data imports"""
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.utils import resample

from joblib import dump, load
import os
import joblib


DATA = pd.read_csv(r"~/Desktop/DataCoSupplyChainDataset.csv", encoding='cp1252')
SUB_DATA = DATA[DATA['Customer Country'] == 'EE. UU.']

if __name__ == '__main__':
    print('Data set imported')



SUB_DATA['balance'] = [1 if b == 'Shipping canceled' else 0 for b in SUB_DATA['Delivery Status']]

DF_MAJORITY = SUB_DATA[SUB_DATA['balance'] == 0]
DF_MINORITY = SUB_DATA[SUB_DATA['balance'] == 1]

NEW_MAJORITY_NUMBER = ((DF_MINORITY.shape[0]/0.075) - DF_MINORITY.shape[0])
NEW_MAJORITY_NUMBER = int(round(NEW_MAJORITY_NUMBER))

DF_MAJORITY_DOWNSAMPLED = resample(DF_MAJORITY, replace=False, n_samples=NEW_MAJORITY_NUMBER,
                                   random_state=29)


DF_DOWNSAMPLED = pd.concat([DF_MAJORITY_DOWNSAMPLED, DF_MINORITY])



"""Categorize items into 3 major categories: electronics, apparel, sports"""
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

def cat_buckets(Product):
    """Function to categorize the string inputs. This simplification to 3 major levels makes
       this attribute much
       easier to interpret when used in the final model.

    Arguments:
        - String 'Category Name'

    Returns:
        - String categorization into 1 of 3 major buckets: Electronics, Apparel, Sports
    """
    if Product in ELECTRONICS:
        return 'Electronics'
    elif Product in APPAREL:
        return 'Apparel'
    elif Product in SPORTS:
        return 'Sports'
    else:
        return 'Other'

DF_DOWNSAMPLED['Category Buckets'] = DF_DOWNSAMPLED['Category Name'].apply(lambda x: cat_buckets(x))


"""Create time attributes"""
CURRENT_DATE_FORMAT = pd.to_datetime(DF_DOWNSAMPLED['order date (DateOrders)'],
                                     format='%m/%d/%Y %H:%M')

DF_DOWNSAMPLED['date'] = CURRENT_DATE_FORMAT.apply(lambda x: x.strftime('%Y-%m-%d'))
DF_DOWNSAMPLED['month-year'] = CURRENT_DATE_FORMAT.apply(lambda x: x.strftime('%m-%Y'))
DF_DOWNSAMPLED['by-month'] = CURRENT_DATE_FORMAT.apply(lambda x: x.strftime('%m'))
DF_DOWNSAMPLED['by-year'] = CURRENT_DATE_FORMAT.apply(lambda x: x.strftime('%Y'))
DF_DOWNSAMPLED['by-week'] = CURRENT_DATE_FORMAT.apply(lambda x: x.strftime('%V'))
DF_DOWNSAMPLED['week-date'] = CURRENT_DATE_FORMAT.apply(lambda x: x.strftime('%u'))
DF_DOWNSAMPLED['by-date'] = CURRENT_DATE_FORMAT.apply(lambda x: x.strftime('%d'))

DF_DOWNSAMPLED = DF_DOWNSAMPLED.loc[(DF_DOWNSAMPLED['month-year'] != '10-2017') &
                                    (DF_DOWNSAMPLED['month-year'] != '11-2017') &
                                    (DF_DOWNSAMPLED['month-year'] != '12-2017') &
                                    (DF_DOWNSAMPLED['month-year'] != '01-2018')]

def region(state):
    """Function to categorize the state inputs. This simplification to major region
       levels makes this attribute much
       easier to work with when used in the final model.

        Arguments:
            - String state
        Returns:
            - Region levels
    """
    if state in ('CA', 'OR', 'WA'):
        return 'pacific'
    elif state  in  ('AZ', 'NM', 'CO', 'UT', 'NV', 'ID', 'WY', 'MT'):
        return 'mountain'
    elif state in ('ND', 'MN', 'SD', 'IA', 'NE', 'MO', 'KS'):
        return 'west north central'
    elif state in ('OK', 'TX', 'AR', 'LA'):
        return 'west south central'
    elif state in ('WI', 'MI', 'IL', 'IN', 'OH'):
        return 'east north central'
    elif state in ('KY', 'TN', 'MS', 'AL'):
        return 'east south central'
    elif state in ('NY', 'PA', 'NJ'):
        return 'middle atlantic'
    elif state in ('WV', 'MD', 'DE', 'VA', 'NC', 'SC', 'GA', 'FL', 'DC'):
        return 'south atlantic'
    elif state in ('ME', 'VT', 'NH', 'MA', 'CT', 'RI'):
        return 'new england'
    elif state in ('AK', 'HI'):
        return 'pacific'
    else:
        return 99

DF_DOWNSAMPLED['customer-region'] = DF_DOWNSAMPLED['Customer State'].apply(lambda x: region(x))


def cancelled(order):
    """Function to classify whether an order was cancelled or not.

        Arguments:
            -Order status
        Returns:
            -One-hot encoding flag: 1, means order was cancelled; otherwise, 0.
    """
    if order == 'Shipping canceled':
        return 1
    else:
        return 0

DF_DOWNSAMPLED['cancelled'] = DF_DOWNSAMPLED['Delivery Status'].apply(lambda k: cancelled(k))




"""Let's focus on order attributes important to us."""
SUB_DATA_FEATURES = DF_DOWNSAMPLED.loc[:, ['Customer Segment', 'Order Item Quantity',
                                           'Category Buckets', 'week-date', 'customer-region',
                                           'cancelled']]




"""Split data to training/testing"""
X = SUB_DATA_FEATURES.iloc[:, [0, 1, 2, 3, 4]]
Y = SUB_DATA_FEATURES.iloc[:, [-1]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)




"""Get training data & estabilsh base levels"""
X_TRAIN = pd.get_dummies(x_train, columns=['Customer Segment', 'Order Item Quantity',
                                           'Category Buckets', 'week-date', 'customer-region'])
Y_TRAIN = y_train

train_features_x = X_TRAIN.loc[:, ['Customer Segment_Corporate', 'Customer Segment_Home Office',
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
train_y = Y_TRAIN


"""Get test data & establish test levels"""
X_TEST = pd.get_dummies(x_test, columns=['Customer Segment', 'Order Item Quantity',
                                         'Category Buckets', 'week-date', 'customer-region'])
Y_TEST = y_test

test_features_x = X_TEST.loc[:, ['Customer Segment_Corporate', 'Customer Segment_Home Office',
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
test_y = Y_TEST



"""Predefine initial model or load most recent model"""

filename = 'lr_model.sav'

if os.path.isfile('lr_model.sav'):
    current_model = load(filename)
else:
    current_model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.75,
                                       class_weight='balanced', C=0.5)



"""Fit our training data"""
current_model.fit(train_features_x, train_y)



"""Identify score of current model."""
y_pred_train_probs = pd.DataFrame(current_model.predict_proba(test_features_x))
Model1_score = roc_auc_score(y_true=test_y, y_score=y_pred_train_probs.iloc[:, 1],
                             multi_class='ovo')




"""Consider updating the model."""
lr_alt = LogisticRegression()

parameters = {
    'C': [0.2, .33, 0.5, 0.66, 0.75, 1, 1.5, 5, 10, 20],
    'class_weight': ['balanced'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'l1_ratio': [0.2, .33, 0.5, 0.66, 0.75]}

new_model = RandomizedSearchCV(lr_alt, parameters, cv=4, n_iter=15)
new_model.fit(train_features_x, train_y)

y_pred_train_probs = pd.DataFrame(new_model.predict_proba(test_features_x))
Model2_score = roc_auc_score(y_true=test_y, y_score=y_pred_train_probs.iloc[:, 1],
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
Model_score_diff = (Model2_score - Model1_score)


if Model_score_diff <= 0:
    dump(current_model, filename)
elif Model_score_diff < 0.01:
    dump(current_model, filename)
elif Model_score_diff > 0.3:
    dump(current_model, filename)
elif Model2_score >= 0.9:
    dump(current_model, filename)
else:
    dump(new_model, filename)
