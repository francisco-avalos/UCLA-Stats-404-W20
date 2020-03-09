#!/usr/bin/env python
# coding: utf-8

# In[1413]:


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


data = pd.read_csv(r"~/Desktop/DataCoSupplyChainDataset.csv", encoding='cp1252')
sub_data = data[data['Customer Country']=='EE. UU.']


# In[1414]:


"""Downsample non-cancelled orders to 7.5% industry average"""
sub_data['balance'] = [1 if b == 'Shipping canceled' else 0 for b in sub_data['Delivery Status']]

df_majority = sub_data[sub_data['balance']==0]
df_minority = sub_data[sub_data['balance']==1]

new_majority_number = ((df_minority.shape[0]/0.075) - df_minority.shape[0])
new_majority_number = int(round(new_majority_number))

df_majority_downsampled = resample(df_majority, replace=False, n_samples=new_majority_number, random_state=29) 

df_downsampled = pd.concat([df_majority_downsampled, df_minority])


# In[1415]:


"""Categorize items into 3 major categories: electronics, apparel, sports"""
Electronics = ['Electronics','Music', 'DVDs', 'Video Games', 'CDs ', 'Consumer Electronics', 'Cameras ', 'Computers']
Apparel = ["Girls' Apparel", "Women's Apparel", "Women's Clothing", "Men's Footwear", 
                  "Men's Clothing", "Children's Clothing", 'Baby ', 'Health and Beauty']
Sports = ['Sporting Goods', 'Cardio Equipment', 'Cleats', 'Shop By Sport', 'Hunting & Shooting', 
              'Tennis & Racquet', 'Baseball & Softball', 'Fitness Accessories', 'Golf Balls', 'Lacrosse', 
              'Boxing & MMA', 'Soccer', 'Fishing', 'Camping & Hiking', 'Hockey', 'Basketball', 'Strength Training', 
              'Golf Gloves', 'Golf Bags & Carts', 'Golf Shoes', 'Golf Apparel', "Women's Golf Clubs", 
              "Men's Golf Clubs", 'Water Sports', 'Indoor/Outdoor Games', "Kids' Golf Clubs", 'Toys', 'As Seen on  TV!', 
              'Accessories', 'Trade-In']

def cat_buckets(Product):
    """Function to categorize the string inputs. This simplification to 3 major levels makes this attribute much 
    easier to interpret when used in the final model.
    
    Arguments: 
        - String 'Category Name'
    
    Returns:
        - String categorization into 1 of 3 major buckets: Electronics, Apparel, Sports
    """
    if Product in Electronics:
        return 'Electronics'
    elif Product in Apparel:
        return 'Apparel'
    elif Product in Sports:
        return 'Sports'
    else:
        return 'Other'

df_downsampled['Category Buckets'] = df_downsampled['Category Name'].apply(lambda x: cat_buckets(x)) 


"""Create time attributes"""
current_date_format = pd.to_datetime(df_downsampled['order date (DateOrders)'], format='%m/%d/%Y %H:%M')
df_downsampled['date'] = current_date_format.apply(lambda x: x.strftime('%Y-%m-%d'))
df_downsampled['month-year'] = current_date_format.apply(lambda x: x.strftime('%m-%Y'))
df_downsampled['by-month'] = current_date_format.apply(lambda x: x.strftime('%m'))
df_downsampled['by-year'] = current_date_format.apply(lambda x: x.strftime('%Y'))
df_downsampled['by-week'] = current_date_format.apply(lambda x: x.strftime('%V'))
df_downsampled['week-date'] = current_date_format.apply(lambda x: x.strftime('%u'))
df_downsampled['by-date'] = current_date_format.apply(lambda x: x.strftime('%d'))
df_downsampled = df_downsampled.loc[(df_downsampled['month-year']!='10-2017') & (df_downsampled['month-year']!='11-2017') & 
            (df_downsampled['month-year']!='12-2017') & (df_downsampled['month-year']!='01-2018')]

def region(state):
    """Function to categorize the state inputs. This simplification to major region levels makes this attribute much 
    easier to work with when used in the final model.
        
        Arguments: 
            - String state
        Returns:
            - Region levels
    """
    if (state in ('CA', 'OR', 'WA')):
        return 'pacific'
    elif (state  in  ('AZ', 'NM', 'CO', 'UT', 'NV', 'ID', 'WY', 'MT')):
        return 'mountain'
    elif (state in ('ND', 'MN', 'SD', 'IA', 'NE', 'MO', 'KS')):
        return 'west north central'
    elif (state in ('OK', 'TX', 'AR', 'LA')):
        return 'west south central'
    elif (state in ('WI', 'MI', 'IL', 'IN', 'OH')):
        return 'east north central'
    elif (state in ('KY', 'TN', 'MS', 'AL')):
        return 'east south central'
    elif (state in ('NY', 'PA', 'NJ')):
        return 'middle atlantic'
    elif (state in ('WV', 'MD', 'DE', 'VA', 'NC', 'SC', 'GA', 'FL', 'DC')):
        return 'south atlantic'
    elif (state in ('ME', 'VT', 'NH', 'MA', 'CT', 'RI')):
        return 'new england'
    elif (state in ('AK', 'HI')):
        return 'pacific'
    else:
        return 99
        
df_downsampled['customer-region'] = df_downsampled['Customer State'].apply(lambda x: region(x))


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

df_downsampled['cancelled'] = df_downsampled['Delivery Status'].apply(lambda k: cancelled(k))


# In[1416]:


"""Let's focus on order attributes important to us."""
sub_data_features = df_downsampled.loc[:, ['Customer Segment','Order Item Quantity','Category Buckets',
                                    'week-date', 'customer-region', 'cancelled']]


# In[1417]:


# x_train, x_test, y_train, y_test
# X_TRAIN, Y_TRAIN, train_features_x, train_y
# X_TEST, Y_TEST, test_features_x, test_y


# In[1418]:


"""Split data to training/testing"""
X = sub_data_features.iloc[:, [0,1,2,3,4]]
Y = sub_data_features.iloc[:, [-1]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


# In[1419]:


"""Get training data & estabilsh base levels"""
X_TRAIN = pd.get_dummies(x_train, columns=['Customer Segment', 'Order Item Quantity', 'Category Buckets',
                                 'week-date', 'customer-region'])
Y_TRAIN = y_train

train_features_x = X_TRAIN.loc[:, ['Customer Segment_Corporate','Customer Segment_Home Office',
       'Order Item Quantity_2', 'Order Item Quantity_3',
       'Order Item Quantity_4', 'Order Item Quantity_5',
       'Category Buckets_Electronics','Category Buckets_Sports', 
       'week-date_2', 'week-date_3','week-date_4', 'week-date_5', 'week-date_6', 'week-date_7',
       'customer-region_east north central','customer-region_east south central', 
       'customer-region_middle atlantic', 'customer-region_mountain', 'customer-region_new england', 
       'customer-region_south atlantic','customer-region_west north central','customer-region_west south central']
]
train_y = Y_TRAIN


"""Get test data & establish test levels"""
X_TEST = pd.get_dummies(x_test, columns=['Customer Segment', 'Order Item Quantity', 'Category Buckets',
                                 'week-date', 'customer-region'])
Y_TEST = y_test

test_features_x = X_TEST.loc[:, ['Customer Segment_Corporate','Customer Segment_Home Office',
       'Order Item Quantity_2', 'Order Item Quantity_3',
       'Order Item Quantity_4', 'Order Item Quantity_5',
       'Category Buckets_Electronics','Category Buckets_Sports', 
       'week-date_2', 'week-date_3','week-date_4', 'week-date_5', 'week-date_6', 'week-date_7',
       'customer-region_east north central','customer-region_east south central', 
       'customer-region_middle atlantic', 'customer-region_mountain', 'customer-region_new england', 
       'customer-region_south atlantic','customer-region_west north central','customer-region_west south central']
]
test_y = Y_TEST


# In[1421]:


"""Predefine initial model or load most recent model"""

filename = 'lr_model.sav'

if os.path.isfile('lr_model.sav'):
    current_model = load(filename)
else:
    current_model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.75, 
                                       class_weight='balanced', C=0.5)    


# In[1442]:


"""Fit our training data"""
current_model.fit(train_features_x, train_y)


# In[1424]:


"""Identify score of current model."""
y_pred_train_probs = pd.DataFrame(current_model.predict_proba(test_features_x))
Model1_score = roc_auc_score(y_true=test_y,y_score=y_pred_train_probs.iloc[:, 1], multi_class='ovo')


# In[1443]:


"""Consider updating the model."""
lr_alt = LogisticRegression()

parameters = {
    'C': [0.2,.33,0.5,0.66,0.75,1,1.5,5,10,20],
    'class_weight': ['balanced'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'l1_ratio': [0.2,.33,0.5,0.66,0.75]
}

new_model = RandomizedSearchCV(lr_alt, parameters, cv=4, n_iter=10)
new_model.fit(train_features_x, train_y)

y_pred_train_probs = pd.DataFrame(new_model.predict_proba(test_features_x))
Model2_score = roc_auc_score(y_true=test_y,y_score=y_pred_train_probs.iloc[:, 1], multi_class='ovo')


# In[1444]:


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


# # Connecting to AWS

# In[1430]:


# import boto3
# import joblib
# import s3fs


# In[1431]:


# s3 = boto3.resource('s3')
# s3_fs = s3fs.S3FileSystem(anon=True)


# In[1432]:


# for bucket in s3.buckets.all():
#     print(bucket.name)


# In[1433]:


# for file in s3.Bucket('francisco-avalos-bucket').objects.all():
#     print(file.key)


# ## Upload Data CSV to S3 Bucket

# In[1434]:


# # file_name = '/Users/franciscoavalosjr/Desktop/DescriptionDataCoSupplyChain.csv'
# df = pd.read_csv('/Users/franciscoavalosjr/Desktop/DataCoSupplyChainDataset.csv', encoding='cp1252')
# # df.head()
# bucket_name = 'francisco-avalos-bucket'
# key_name = 'DataSupplyChainData'


# In[1435]:


# s3_fs = s3fs.S3FileSystem(anon=False)


# In[1436]:


# with s3_fs.open(f"{bucket_name}/{key_name}", "w") as file: 
#     df.to_csv(file)


# ## Upload Model to S3 Bucket

# In[1437]:


# model = joblib.load("/Users/franciscoavalosjr/Desktop/Side_Projects/lr.joblib")


# In[1438]:


# key_name = "project_model.joblib"


# In[1439]:


# with s3_fs.open(f"{bucket_name}/{key_name}","wb") as file:
#     joblib.dump('project-model', model)

