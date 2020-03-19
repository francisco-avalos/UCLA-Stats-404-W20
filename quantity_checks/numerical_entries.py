import pandas as pd


def quantity_entries(order_quantity):
    """The function receives an int quantity
       and validates whether it's good for use.
    """
    if (order_quantity <= 0) | (order_quantity > 5):
        raise ValueError('Error: Orders must be between 1 and 5.')
    elif isinstance(order_quantity, int):
        return order_quantity
    else:
        raise ValueError('Error: entry must be of type int')


def date_entries(date_input):
    """The function checks that the date entered is valid; if it is, it returns the numerical day of the week; otherwise
       it returns the issue.
    """
    if not isinstance(date_input, str):
        raise ValueError('Error: Please enter date as a string as "MM/DD/YYYY HH:MM" format')

    date_entered = pd.to_datetime(date_input, format='%m/%d/%Y %H:%M')

    date_entered = date_entered.strftime('%u')
    date_entered = int(date_entered)
    return date_entered


def client_entries(type_of_client):
    """Function determines whether entry is of
        an appropriate customer type
    """
    type_of_client.lower()
    if type_of_client not in ('customer', 'business', 'home office'):
        raise ValueError('Error: Must be entry of type customer, business, home office')
    else:
        return type_of_client


def product_entry(product_input):
    """Function determines whether item purchased is valid"""
    product_input.lower()
    if product_input not in ('apparel', 'electronics', 'sports'):
        raise ValueError('Error: input must be electronics or sports')
    else:
        return product_input


def region_entry(region_input):
    """Function determines whether entry is a valid region entry"""
    region_input.lower()
    if region_input not in ('pacific', 'mountain', 'west north central', 'east north central', 'west south central',
                            'east south central', 'south atlantic', 'middle atlantic', 'new england'):
        return ValueError('Error: input must be one of 9 region inputs')
    else:
        return region_input
