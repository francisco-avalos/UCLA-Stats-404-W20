import sys
import numpy as np
import pandas as pd
import urllib.request
import logging
import joblib
from tests.numerical_entries import quantity_entries, date_entries, client_entries, product_entry, \
    region_entry

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def import_model():
    """Import the model from AWS S3 Bucket"""
    s3_url = 'https://francisco-avalos-bucket.s3-us-west-2.amazonaws.com/lr_model.sav'
    s3_model = joblib.load(urllib.request.urlopen(s3_url))
    return s3_model


if __name__ == '__main__':

    LOGGER.info("Import model from AWS")
    logistic_model = import_model()
    LOGGER.info("Imported model from AWS successful")

    CUSTOMER = str(sys.argv[1])
    PRODUCT = str(sys.argv[2])
    DAY_OF_WEEK = str(sys.argv[3])
    REGION = str(sys.argv[4])
    QUANTITY = int(sys.argv[5])

    LOGGER.info("Entries read and verified")
    CUSTOMER = client_entries(CUSTOMER)
    QUANTITY = quantity_entries(QUANTITY)
    DAY_OF_WEEK = date_entries(DAY_OF_WEEK)
    PRODUCT = product_entry(PRODUCT)
    REGION = region_entry(REGION)

    print('You\'ve entered\n')
    print(f"customer: {CUSTOMER}")
    print(f"quantity: {QUANTITY}")
    print(f"product: {PRODUCT}")
    print(f"day of week: {DAY_OF_WEEK}")
    print(f"region: {REGION}")

    LOGGER.info("Encoding entries")
    CUSTOMER_ENTRY = [] * 2
    if CUSTOMER == 'customer':
        CUSTOMER_ENTRY = [0, 0]
    elif CUSTOMER == 'business':
        CUSTOMER_ENTRY = [1, 0]
    elif CUSTOMER == 'home office':
        CUSTOMER_ENTRY = [0, 1]

    QUANTITY_ENTRY = [] * 4
    if QUANTITY == 1:
        QUANTITY_ENTRY = [0, 0, 0, 0]
    elif QUANTITY == 2:
        QUANTITY_ENTRY = [1, 0, 0, 0]
    elif QUANTITY == 3:
        QUANTITY_ENTRY = [0, 1, 0, 0]
    elif QUANTITY == 4:
        QUANTITY_ENTRY = [0, 0, 1, 0]
    elif QUANTITY == 5:
        QUANTITY_ENTRY = [0, 0, 0, 1]

    DAY_ENTRY = [] * 6
    if DAY_OF_WEEK == 1:
        DAY_ENTRY = [0, 0, 0, 0, 0, 0]
    elif DAY_OF_WEEK == 2:
        DAY_ENTRY = [1, 0, 0, 0, 0, 0]
    elif DAY_OF_WEEK == 3:
        DAY_ENTRY = [0, 1, 0, 0, 0, 0]
    elif DAY_OF_WEEK == 4:
        DAY_ENTRY = [0, 0, 1, 0, 0, 0]
    elif DAY_OF_WEEK == 5:
        DAY_ENTRY = [0, 0, 0, 1, 0, 0]
    elif DAY_OF_WEEK == 6:
        DAY_ENTRY = [0, 0, 0, 0, 1, 0]
    elif DAY_OF_WEEK == 7:
        DAY_ENTRY = [0, 0, 0, 0, 0, 1]

    PRODUCT_ENTRY = [] * 2
    if PRODUCT == 'apparel':
        PRODUCT_ENTRY = [0, 0]
    elif PRODUCT == 'electronics':
        PRODUCT_ENTRY = [1, 0]
    elif PRODUCT == 'sports':
        PRODUCT_ENTRY = [0, 1]

    REGION_ENTRY = [] * 8
    if REGION == 'pacific':
        REGION_ENTRY = [0, 0, 0, 0, 0, 0, 0, 0]
    elif REGION == 'east north central':
        REGION_ENTRY = [1, 0, 0, 0, 0, 0, 0, 0]
    elif REGION == 'east south central':
        REGION_ENTRY = [0, 1, 0, 0, 0, 0, 0, 0]
    elif REGION == 'middle atlantic':
        REGION_ENTRY = [0, 0, 1, 0, 0, 0, 0, 0]
    elif REGION == 'mountain':
        REGION_ENTRY = [0, 0, 0, 1, 0, 0, 0, 0]
    elif REGION == 'new england':
        REGION_ENTRY = [0, 0, 0, 0, 1, 0, 0, 0]
    elif REGION == 'south atlantic':
        REGION_ENTRY = [0, 0, 0, 0, 0, 1, 0, 0]
    elif REGION == 'west north central':
        REGION_ENTRY = [0, 0, 0, 0, 0, 0, 1, 0]
    elif REGION == 'west south central':
        REGION_ENTRY = [0, 0, 0, 0, 0, 0, 0, 1]

    QUAN2_DAY4_INTERACTION = [] * 1
    if (QUANTITY == 2) & (DAY_OF_WEEK == 4):
        QUAN2_DAY4_INTERACTION = [1]
    else:
        QUAN2_DAY4_INTERACTION = [0]

    QUAN3_DAY4_INTERACTION = [] * 1
    if (QUANTITY == 3) & (DAY_OF_WEEK == 4):
        QUAN3_DAY4_INTERACTION = [1]
    else:
        QUAN3_DAY4_INTERACTION = [0]

    QUAN2_DAY5_INTERACTION = [] * 1
    if (QUANTITY == 2) & (DAY_OF_WEEK == 5):
        QUAN2_DAY5_INTERACTION = [1]
    else:
        QUAN2_DAY5_INTERACTION = [0]

    QUAN3_DAY5_INTERACTION = [] * 1
    if (QUANTITY == 3) & (DAY_OF_WEEK == 5):
        QUAN3_DAY5_INTERACTION = [1]
    else:
        QUAN3_DAY5_INTERACTION = [0]

    REG_ENC_PROD_ELEC_INTERACTION = [] * 1
    if (REGION == 'east north central') & (PRODUCT == 'electronics'):
        REG_ENC_PROD_ELEC_INTERACTION = [1]
    else:
        REG_ENC_PROD_ELEC_INTERACTION = [0]

    CUS_HO_QUAN2_INTERACTION = [] * 1
    if (CUSTOMER == 'home office') & (QUANTITY == 2):
        CUS_HO_QUAN2_INTERACTION = [1]
    else:
        CUS_HO_QUAN2_INTERACTION = [0]

    X_Entered = np.concatenate((CUSTOMER_ENTRY, PRODUCT_ENTRY, DAY_ENTRY, REGION_ENTRY, QUANTITY_ENTRY,
                                QUAN2_DAY4_INTERACTION, QUAN3_DAY4_INTERACTION, QUAN2_DAY5_INTERACTION,
                                QUAN3_DAY5_INTERACTION, REG_ENC_PROD_ELEC_INTERACTION, CUS_HO_QUAN2_INTERACTION))
    X_Entered = pd.DataFrame(X_Entered)
    X_Entered = np.transpose(X_Entered)

    LOGGER.info("Generating prediction (0, order likely to NOT cancel; 1, order likely to cancel)")
    print(logistic_model.predict(X_Entered))

