import main
import sys
import os
from joblib import dump, load

# FILENAME = 'lr_model.sav'

# if os.path.isfile('lr_model.sav'):
# 	CURRENT_MODEL = load(FILENAME)


print('enter something')
if __name__ == '__main__':
    CUSTOMER_TYPE = sys.argv[1]

    print(f"printing your input: {CUSTOMER_TYPE}")