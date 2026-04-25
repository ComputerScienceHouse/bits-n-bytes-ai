###############################################################################
#
# File: database.py
#
# Author: Isaac Ingram
#
# Purpose: Provide a connection to the database
#
###############################################################################
import os
import requests
from typing import List
import config
from data_classes import *

API_ENDPOINT = os.getenv("BNB_API_ENDPOINT", '')
AUTHORIZATION_KEY = os.getenv("BNB_AUTHORIZATION_KEY", '')

MOCK_ITEMS = {
    2: Item(2, "Sour Patch Kids", "567890123456", 3.50, 90, 200, 20, "images/item_placeholder.png", ""),
    3: Item(3, "Brownie Brittle", "", 3.00, 90, 78, 10, "images/item_placeholder.png", ""),
}

# [
#     {
#         "id": 1,
#         "name": "Jolt Soda",
#         "price": 1.5,
#         "quantity": 1,
#         "thumb_img": "http://placehold.jp/150x150.png",
#         "upc": "1000000000",
#         "vision_class": "",
#         "weight_avg": 1.0,
#         "weight_std": 1.0
#     },
#     {
#         "id": 2,
#         "name": "Sour Patch Kids",
#         "price": 2.5,
#         "quantity": 1,
#         "thumb_img": "http://placehold.jp/150x150.png",
#         "upc": "070462035964",
#         "vision_class": "",
#         "weight_avg": 226.0,
#         "weight_std": 10.0
#     },
#     {
#         "id": 3,
#         "name": "Brownie Brittle",
#         "price": 2.5,
#         "quantity": 1,
#         "thumb_img": "http://placehold.jp/150x150.png",
#         "upc": "711747011128",
#         "vision_class": "",
#         "weight_avg": 78.0,
#         "weight_std": 10.0
#     },
#     {
#         "id": 4,
#         "name": "Little Bites Blueberry",
#         "price": 2.1,
#         "quantity": 1,
#         "thumb_img": "http://placehold.jp/150x150.png",
#         "upc": "072030013398",
#         "vision_class": "",
#         "weight_avg": 47.0,
#         "weight_std": 10.0
#     },
#     {
#         "id": 5,
#         "name": "Pepsi Wild Cherry 12 Pack",
#         "price": 8.8,
#         "quantity": 1,
#         "thumb_img": "http://placehold.jp/150x150.png",
#         "upc": "012000809996",
#         "vision_class": "",
#         "weight_avg": 4082.33,
#         "weight_std": 20.0
#     }
# ]

# Use mock data if USE_MOCK_DATA environment variable is set to 'true'. If it
# isn't set to 'true' (including not being set at all), it this defaults to
# False.
USE_MOCK_DB_DATA = os.getenv("USE_MOCK_DB_DATA", 'false').lower() == 'true'

REQUEST_HEADERS = {"Authorization": AUTHORIZATION_KEY}

# Store a cached list of all items
cached_items = None
# Store a cached dictionary of items, accessible by ID
cached_items_by_id = None


def is_reachable() -> bool:
    """
    Check if the database is reachable
    :return: True if the database is reachable, False otherwise
    """
    if USE_MOCK_DB_DATA:
        return True
    else:
        print("Check If Reachable (GET)")
        try:
            requests.get(API_ENDPOINT, headers=REQUEST_HEADERS)
            return True
        except requests.RequestException:
            print(f"\tExperienced Request Exception")
            return False


def get_items() -> List[Item]:
    """
    Get all items
    :return: A List of Item. If there is an error, an empty list is returned
    """
    global cached_items, cached_items_by_id

    print("GET /get_items")
    if USE_MOCK_DB_DATA:
        return list(MOCK_ITEMS.values())
    else:
        # Fetch data only if there is no cache
        if cached_items is None:
            url = API_ENDPOINT + "get_items"
            # Make request
            response = requests.get(url, headers={"Authorization": AUTHORIZATION_KEY})
            # Check response code
            if response.status_code == 200:
                # Create list of items
                result = list()
                for item_raw in response.json():
                    result.append(Item(
                        item_raw['id'],
                        item_raw['name'],
                        item_raw['upc'],
                        item_raw['price'],
                        item_raw['quantity'],
                        item_raw['weight_avg'],
                        item_raw['weight_std'],
                        item_raw['thumb_img'],
                        item_raw['vision_class']
                    ))
                # Cache the items list
                cached_items = result
                # Cache items by ID
                cached_items_by_id = dict()
                for item in result:
                    cached_items_by_id[item.item_id] = item
                    print(f"stored item {item.item_id} in cache")
                return result
            else:
                # Something went wrong so print info and return empty list
                print(f"\tReceived response {response.status_code}:")
                print(f"\t{response.content}")
                return list()
        else:
            return cached_items


def get_item(item_id: int) -> Item | None:
    """
    Get an item from its ID
    :return: An Item or None if the item does not exist
    """
    global cached_items_by_id
    print(f"GET /items/{item_id}")
    if USE_MOCK_DB_DATA:
        if item_id in MOCK_ITEMS:
            return MOCK_ITEMS[item_id]
        else:
            return None
    else:
        if cached_items_by_id is None:
            url = API_ENDPOINT + f"/items/{item_id}"
            # Make request
            response = requests.get(url, headers=REQUEST_HEADERS)
            # Check response code
            if response.status_code == 200:
                item = response.json()
                return Item(
                    item['id'],
                    item['name'],
                    item['upc'],
                    item['price'],
                    item['quantity'],
                    item['weight_avg'],
                    item['weight_std'],
                    item['thumb_img'],
                    item['vision_class']
                )
            else:
                # Something went wrong so print info and return None
                print(f"\tReceived response {response.status_code}:")
                print(f"\t{response.content}")
                return None
        else:
            return cached_items_by_id[item_id]


def get_shelf_contents(shelf_id: str) -> list:
    """
    Get all slot contents for a shelf.
    :return: List of {slot_id, items: [{id, quantity, ...}]} or empty list on error
    """
    print(f"GET /shelf/{shelf_id}")
    if USE_MOCK_DB_DATA:
        return []
    url = API_ENDPOINT + f"shelf/{shelf_id}"
    try:
        response = requests.get(url, headers=REQUEST_HEADERS)
        if response.status_code == 200:
            return response.json()
        print(f"\tReceived response {response.status_code}: {response.content}")
        return []
    except requests.RequestException as e:
        print(f"\tRequest exception: {e}")
        return []


def update_shelf_slot_quantity(shelf_id: str, slot_id: int, item_id: int, quantity: int) -> bool:
    """
    Update the quantity of an item in a shelf slot. PUT /shelf/<shelf_id>/slot/<slot_id>
    :return: True on success, False on failure
    """
    print(f"PUT /shelf/{shelf_id}/slot/{slot_id} item_id={item_id} quantity={quantity}")
    if USE_MOCK_DB_DATA:
        return True
    url = API_ENDPOINT + f"shelf/{shelf_id}/slot/{slot_id}"
    try:
        response = requests.put(url, json={'item_id': item_id, 'quantity': quantity}, headers=REQUEST_HEADERS)
        if response.status_code == 200:
            return True
        print(f"\tReceived response {response.status_code}: {response.content}")
        return False
    except requests.RequestException as e:
        print(f"\tRequest exception: {e}")
        return False


def add_shelf_slot_item(shelf_id: str, slot_id: int, item_id: int, quantity: int) -> bool:
    """
    Add an item to a shelf slot. POST /shelf/<shelf_id>/slot/<slot_id>
    :return: True on success, False on failure
    """
    print(f"POST /shelf/{shelf_id}/slot/{slot_id} item_id={item_id} quantity={quantity}")
    if USE_MOCK_DB_DATA:
        return True
    url = API_ENDPOINT + f"shelf/{shelf_id}/slot/{slot_id}"
    try:
        response = requests.post(url, json={'item_id': item_id, 'quantity': quantity}, headers=REQUEST_HEADERS)
        if response.status_code in (200, 201):
            return True
        print(f"\tReceived response {response.status_code}: {response.content}")
        return False
    except requests.RequestException as e:
        print(f"\tRequest exception: {e}")
        return False


def remove_shelf_slot_item(shelf_id: str, slot_id: int, item_id: int) -> bool:
    """
    Remove an item from a shelf slot. DELETE /shelf/<shelf_id>/slot/<slot_id>
    :return: True on success, False on failure
    """
    print(f"DELETE /shelf/{shelf_id}/slot/{slot_id} item_id={item_id}")
    if USE_MOCK_DB_DATA:
        return True
    url = API_ENDPOINT + f"shelf/{shelf_id}/slot/{slot_id}"
    try:
        response = requests.delete(url, json={'item_id': item_id}, headers=REQUEST_HEADERS)
        if response.status_code in (200, 204):
            return True
        print(f"\tReceived response {response.status_code}: {response.content}")
        return False
    except requests.RequestException as e:
        print(f"\tRequest exception: {e}")
        return False
