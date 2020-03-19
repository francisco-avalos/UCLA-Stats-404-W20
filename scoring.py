import sys
import numpy as np
import pandas as pd


from quantity_checks.numerical_entries import quantity_entries, date_entries, client_entries, product_entry, \
    region_entry
from joblib import load

FILENAME = 'lr_model.sav'
CURRENT_MODEL = load(FILENAME)

# print(CURRENT_MODEL)


if __name__ == '__main__':
    # --- Read-in items from command line;

    CUSTOMER = str(sys.argv[1])
    PRODUCT = str(sys.argv[2])
    DAY_OF_WEEK = str(sys.argv[3])
    REGION = str(sys.argv[4])
    QUANTITY = int(sys.argv[5])

    # --- Print them out, for validation:
    # print('You\'ve entered\n')
    # print(f"customer: {CUSTOMER}")
    # print(f"quantity: {QUANTITY}")
    # print(f"product: {PRODUCT}")
    # print(f"day of week: {DAY_OF_WEEK}")
    # print(f"region: {REGION}")

    CUSTOMER = client_entries(CUSTOMER)
    QUANTITY = quantity_entries(QUANTITY)
    DAY_OF_WEEK = date_entries(DAY_OF_WEEK)
    PRODUCT = product_entry(PRODUCT)
    REGION = region_entry(REGION)

    customer_entry = [] * 2
    if CUSTOMER == 'customer':
        customer_entry = [0, 0]
    elif CUSTOMER == 'business':
        customer_entry = [1, 0]
    elif CUSTOMER == 'home office':
        customer_entry = [0, 1]

    quantity_entry = [] * 4
    if QUANTITY == 1:
        quantity_entry = [0, 0, 0, 0]
    elif QUANTITY == 2:
        quantity_entry = [1, 0, 0, 0]
    elif QUANTITY == 3:
        quantity_entry = [0, 1, 0, 0]
    elif QUANTITY == 4:
        quantity_entry = [0, 0, 1, 0]
    elif QUANTITY == 5:
        quantity_entry = [0, 0, 0, 1]

    day_entry = [] * 6
    if DAY_OF_WEEK == 1:
        day_entry = [0, 0, 0, 0, 0, 0]
    elif DAY_OF_WEEK == 2:
        day_entry = [1, 0, 0, 0, 0, 0]
    elif DAY_OF_WEEK == 3:
        day_entry = [0, 1, 0, 0, 0, 0]
    elif DAY_OF_WEEK == 4:
        day_entry = [0, 0, 1, 0, 0, 0]
    elif DAY_OF_WEEK == 5:
        day_entry = [0, 0, 0, 1, 0, 0]
    elif DAY_OF_WEEK == 6:
        day_entry = [0, 0, 0, 0, 1, 0]
    elif DAY_OF_WEEK == 7:
        day_entry = [0, 0, 0, 0, 0, 1]

    product_entry = [] * 2
    if PRODUCT == 'apparel':
        product_entry = [0, 0]
    elif PRODUCT == 'electronics':
        product_entry = [1, 0]
    elif PRODUCT == 'sports':
        product_entry = [0, 1]

    region_entry = [] * 8
    if REGION == 'pacific':
        region_entry = [0, 0, 0, 0, 0, 0, 0, 0]
    elif REGION == 'east north central':
        region_entry = [1, 0, 0, 0, 0, 0, 0, 0]
    elif REGION == 'east south central':
        region_entry = [0, 1, 0, 0, 0, 0, 0, 0]
    elif REGION == 'middle atlantic':
        region_entry = [0, 0, 1, 0, 0, 0, 0, 0]
    elif REGION == 'mountain':
        region_entry = [0, 0, 0, 1, 0, 0, 0, 0]
    elif REGION == 'new england':
        region_entry = [0, 0, 0, 0, 1, 0, 0, 0]
    elif REGION == 'south atlantic':
        region_entry = [0, 0, 0, 0, 0, 1, 0, 0]
    elif REGION == 'west north central':
        region_entry = [0, 0, 0, 0, 0, 0, 1, 0]
    elif REGION == 'west south central':
        region_entry = [0, 0, 0, 0, 0, 0, 0, 1]

    interaction1 = [] * 1
    if (QUANTITY == 2) & (DAY_OF_WEEK == 4):
        interaction1 = [1]
    else:
        interaction1 = [0]

    interaction2 = [] * 1
    if (QUANTITY == 3) & (DAY_OF_WEEK == 4):
        interaction2 = [1]
    else:
        interaction2 = [0]

    interaction3 = [] * 1
    if (QUANTITY == 2) & (DAY_OF_WEEK == 5):
        interaction3 = [1]
    else:
        interaction3 = [0]

    interaction4 = [] * 1
    if (QUANTITY == 3) & (DAY_OF_WEEK == 5):
        interaction4 = [1]
    else:
        interaction4 = [0]

    interaction5 = [] * 1
    if (REGION == 'east north central') & (PRODUCT == 'electronics'):
        interaction5 = [1]
    else:
        interaction5 = [0]

    interaction6 = [] * 1
    if (CUSTOMER == 'home office') & (QUANTITY == 2):
        interaction6 = [1]
    else:
        interaction6 = [0]

    print(f"Customer code: {customer_entry}")
    print(f"product code: {product_entry}")
    print(f"day code: {day_entry}")
    print(f"region code: {region_entry}")
    print(f"quantity code: {quantity_entry}")
    print(f"interaction1 code: {interaction1}")
    print(f"interaction2 code: {interaction2}")
    print(f"interaction3 code: {interaction3}")
    print(f"interaction4 code: {interaction4}")
    print(f"interaction5 code: {interaction5}")
    print(f"interaction6 code: {interaction6}")

    X_Entered = np.concatenate((customer_entry, product_entry, day_entry, region_entry, quantity_entry, interaction1,
                                interaction2, interaction3, interaction4, interaction5, interaction6))
    X_Entered = pd.DataFrame(X_Entered)
    X_Entered = np.transpose(X_Entered)

    print(X_Entered)

    # print(CURRENT_MODEL.predict(X_Entered))
