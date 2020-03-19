import pandas as pd


def quantity_entries(order_quantity):
    """The function receives an int quantity
       and validates whether it's good for use.
    """
    if not isinstance(order_quantity, int):
        ValueError('Error: entry must be an int entry')
    elif (order_quantity <= 0) | (order_quantity > 5):
        raise ValueError('Error: Orders must be between 1 and 5')
    elif isinstance(order_quantity, int):
        return order_quantity


def date_entries(date_input):
    """The function checks that the date entered is valid; if it is, it returns the numerical day of the week; otherwise
       it returns the issue.
    """
    if not isinstance(date_input, str):
        raise ValueError('Error: Please enter date as a string in the "MM/DD/YYYY HH:MM" format')

    date_entered = pd.to_datetime(date_input, format='%m/%d/%Y %H:%M')

    date_entered = date_entered.strftime('%u')
    date_entered = int(date_entered)
    return date_entered


def client_entries(type_of_client):
    """Function determines whether entry is of
        an appropriate customer type
    """
    type_of_client.lower()
    if not isinstance(type_of_client, str):
        raise ValueError('Error: client entry must be string entry')
    elif type_of_client not in ('customer', 'business', 'home office'):
        raise ValueError('Error: Must be entry of type customer, business, home office')
    else:
        return type_of_client


def product_entry(product_input):
    """Function determines whether item purchased is valid"""
    product_input = product_input.lower()
    if not isinstance(product_input, str):
        raise ValueError('Error: client entry must be string entry')
    elif product_input not in ('apparel', 'electronics', 'sports'):
        raise ValueError('Error: input must be: apparel, electronics or sports')
    else:
        return product_input


def region_entry(region_input):
    """Function determines whether entry is a valid region entry"""
    region_input.lower()
    if region_input not in ('pacific', 'mountain', 'west north central', 'east north central', 'west south central',
                            'east south central', 'south atlantic', 'middle atlantic', 'new england'):
        return ValueError('Error: input must be one of 9 region inputs: pacific, mountain, west north central, '
                          'east north central, west south central, east south central, south atlantic, middle atlantic,'
                          'new england')
    else:
        return region_input
