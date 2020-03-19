import pandas
import pytest

from tests.numerical_entries import date_entries, quantity_entries

def test_date_type():
	"""function to test date entries"""
	expected_output = 3
	output = date_entries('3/1/2000 01:10')


def test_quantity_entries():
	expected_output = 5
	output = quantity_entries(1)


def quantity_integration_with_date_entries():
	"""Integration test to check that (reasonably) as the weekened approaches, more items 
		are bought. The purpose here is solely to check the integration between these two 
		functions and not necessarily claim that this assumption is true.
	"""
	expected_output = 5
	day_of_week = date_entries('03/13/2020 05:00')
	items_ordered = quantity_entries(day_of_week)
