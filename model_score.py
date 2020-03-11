# import main
import sys
import os
from joblib import dump, load

# FILENAME = 'lr_model.sav'

# if os.path.isfile('lr_model.sav'):
# 	CURRENT_MODEL = load(FILENAME)


if __name__ == '__main__':
    CUSTOMER_TYPE = str(sys.argv[1])
    QUANTITY = int(sys.argv[2])
    CATEGORY = str(sys.argv[3])
    WEEK_DATE = int(sys.argv[4])
    CUSTOMER_REGION = str(sys.argv[5])

    print(f"printing this: {CUSTOMER_TYPE}")
    print(f"printing this: {QUANTITY}")
    print(f"printing this: {CATEGORY}")
    print(f"printing this: {WEEK_DATE}")
    print(f"printing this: {CUSTOMER_REGION}")

    def customer_type(CUSTOMER_TYPE):
    	CUSTOMER_TYPE.lower()
    	if CUSTOMER_TYPE not in ('customer', 'corporate', 'home-office'):
    		print('Error: Enter a valid entry (customer, corporate, or home-office)')
    	return CUSTOMER_TYPE
    
    def quantity_amount(QUANTITY):
    	if QUANTITY <= 0:
    		print('Error: Enter a int >= 1')
    	return QUANTITY
    def category_type(CATEGORY):
    	CATEGORY.lower()
    	if CATEGORY not in ('electronics', 'sports'):
    		print('Error: Invalid entry (must be apparel, electronics, sports)')
    	return CATEGORY
    def week_date_classifier(WEEK_DATE):
    	if WEEK_DATE not in (2, 3, 4, 5, 6, 7):
    		print('Errror: Invalid entry. Must be between 1-7')
    	return WEEK_DATE
    def specify_region(CUSTOMER_REGION):
    	CUSTOMER_REGION.lower()
    	if CUSTOMER_REGION not in ('mountain', 'west north central', 'east north central',
                                   'west south central', 'east south central', 'south atlantic',
                                   'middle atlantic', 'new england'):
    	    print('Erorr: Invalid entry (must a US region)')
    	return CUSTOMER_REGION

    def convert_customer_type(CUSTOMER_TYPE):
    	if CUSTOMER_REGION == 'apparel':
            customer_list = [0, 0]
    	return customer_list
    
    result = convert_customer_type(CUSTOMER_REGION)
    # print(result)
    # print((convert_customer_type(CUSTOMER_TYPE))


	# print(CURRENT_MODEL)
