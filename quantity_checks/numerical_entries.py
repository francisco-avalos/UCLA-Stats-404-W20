import math
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


def hour_check_date_entry(hour_input):
    """The function checks whether the hour entered is valid"""
    if (hour_input > 0) & (hour_input <= 23):
        return True
    else:
        raise ValueError('Error: hour entry must be between 00 and 23 (inclusive)')


def minute_check_date_entry(minute_input):
    """The function checks whether the minute entered is valid"""
    if (minute_input > 0) & (minute_input <= 59):
        return True
    else:
        raise ValueError("Error: minute entry must be between 00 and 59")


def hour_check_date_entry(date_entered):
    """The function returns the hours for the given entry"""
    if isinstance(date_entered, str):
        date_entered = pd.to_datetime(date_entered, format='%m/%d/%Y %H:%M')
        converted_minute = date_entered.strftime('%M')
        return converted_minute
    else:
        raise ValueError('')


def date_entries(date_input):
    """The function checks that the date entered is valid; if it is, it returns the numerical day of the week; otherwise
       it returns the issue.
    """
    if not isinstance(date_input, str):
        raise ValueError('Error: Please enter date as a string as "MM/DD/YYYY HH:MM" format')

    date_entered = pd.to_datetime(date_input, format='%m/%d/%Y %H:%M')
    # month_entered = date_entered.strftime('%m')
    # hour_entered = date_entered.strftime('%H')
    # minutes_entered = date_entered.strftime('%M')
    #
    # if not hour_check_date_entry(hour_entered):
    #     return hour_check_date_entry(hour_entered)
    # if not minute_check_date_entry(minutes_entered):
    #     return minute_check_date_entry(minutes_entered)

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
