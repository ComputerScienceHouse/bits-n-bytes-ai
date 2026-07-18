#####################################################################
#                                                                   #
# file: cartservice.py                                              #
# author: Akash Keshav                                              #
# purpose: create a shared cart for the two AI services to          #
# interact with.                                                    #
#                                                                   #
#####################################################################

# imports
import json
import threading
import time
from typing import Dict, Tuple

import serial

import database as db
from model import Shelf, Item, Slot
from config import *

# initialize globals
PI_UART_PORT = None
ESP_UART_PORT = None
CLEAR_CART_CMD = bytes([0xDE, 0xAD, 0xBE, 0xEF])

MAC_ADDR_TO_SHELVES: Dict[str, Shelf] = {}
CART_DICT: Dict[int, Tuple[int, str]] = {}   # item_id -> (quantity, source)
CART_ITEM_DATA: Dict[int, Item] = {}          # item_id -> Item object (survives DB reloads)

# initialize privates
_lock = threading.Lock()
_running = False

# load items
CLASS_NAME_TO_ITEM_ID = dict()
for item in db.get_items():
    CLASS_NAME_TO_ITEM_ID[item.item_id] = item.vision_class 

def cart_init() -> None:
    """
    Initializes the cart features. Handles setting the ESP and PI UART connections,
    then starts the background service loop.
    """
    global ESP_UART_PORT
    global PI_UART_PORT
    global _running

    if _running:
        return

    ESP_UART_PORT = serial.Serial(
        port=ESP_SERIAL_PORT,
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )

    PI_UART_PORT = serial.Serial(
        port=PI_SERIAL_PORT,
        baudrate=9600,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )

    _running = True
    threading.Thread(target=_cart_service_worker, daemon=True).start()

    print('Cart Service Initialized!')


def cart_stop() -> None:
    """
    Stops the background cart service loop.
    """
    global _running
    _running = False


def getCart() -> Dict[int, Tuple[int, str]]:
    """
    Returns what is currently in the cart as a Dictionary[Item ID, (Quantity, Source)].
    """
    with _lock:
        return dict(CART_DICT)


def add(item_id: int, quantity: int, source: str = 'vision') -> None:
    """
    Adds to what is currently in the cart. Needs Item ID (int) and Quantity (int).
    """
    with _lock:
        current_qty, _ = CART_DICT.get(item_id, (0, source))
        CART_DICT[item_id] = (current_qty + quantity, source)
        if item_id not in CART_ITEM_DATA:
            CART_ITEM_DATA[item_id] = CLASS_NAME_TO_ITEM_ID[item_id]
        out = {'id': item_id, 'quantity': quantity}
        if PI_UART_PORT:
            PI_UART_PORT.write((json.dumps(out) + "\n").encode('utf-8'))
        else:
            print('UART NOT INITIALIZED, NOT SENDING')
    print(f"[{source}] +{quantity}x item {item_id} -> cart={CART_DICT}")


def remove(item_id: int, quantity: int, source: str = 'vision') -> None:
    """
    Removes from what is currently in the cart (e.g. item put back). Needs
    Item ID (int) and Quantity (int). Thread-safe.
    """
    with _lock:
        current_qty, _ = CART_DICT.get(item_id, (0, source))
        new_qty = max(0, current_qty - quantity)
        if new_qty == 0:
            CART_DICT.pop(item_id, None)
            CART_ITEM_DATA.pop(item_id, None)
        else:
            CART_DICT[item_id] = (new_qty, source)
        out = {'id': item_id, 'quantity': -quantity}
        if PI_UART_PORT:
            PI_UART_PORT.write((json.dumps(out) + "\n").encode('utf-8'))
        else:
            print('UART NOT INITIALIZED, NOT SENDING')
    print(f"[{source}] -{quantity}x item {item_id} -> cart={CART_DICT}")


def _cart_service_worker() -> None:
    """
    Background thread entrypoint. Keeps calling the single-pass loop body
    until cart_stop() is called.
    """
    while _running:
        _cart_service_loop()
        time.sleep(0.01)

def is_vision_enabled() -> bool:
    return True

def _cart_service_loop() -> None:
    """
    Single pass of the loop that handles cart changes coming from the shelf
    hardware over serial. add()/remove() above are called directly from
    detect.py's vision instance; this function handles the weight-sensor path.
    """
    # Check for a clear-cart command from the UI before processing shelf data
    if PI_UART_PORT.in_waiting >= len(CLEAR_CART_CMD):
        incoming = PI_UART_PORT.read(len(CLEAR_CART_CMD))
        if incoming == CLEAR_CART_CMD:
            with _lock:
                CART_DICT.clear()
                CART_ITEM_DATA.clear()
            print("Cart cleared by UI command")

    line = ESP_UART_PORT.readline().decode('utf-8', errors='ignore').strip()

    if not line:
        return

    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        print("Unable to decode JSON", line)
        return

    if 'shelf_mac' not in msg or 'slot_id' not in msg or 'delta_g' not in msg:
        print("JSON missing shelf_mac, slot_id, or delta_g")
        return

    mac_address = msg['shelf_mac']
    slot_id = msg['slot_id']
    weight_delta = msg['delta_g']

    if mac_address not in MAC_ADDR_TO_SHELVES:
        MAC_ADDR_TO_SHELVES[mac_address] = Shelf(mac_address)

    shelf = MAC_ADDR_TO_SHELVES[mac_address]
    shelf._load_from_db()
    slot = shelf.get_slot(slot_id)
    time_str = time.strftime("%H:%M:%S.") + f"{int((time.time() * 1000) % 1000):03d}"

    if weight_delta < 0:
        # Item removed from shelf — predict from all items
        item_changes = slot.predict_most_likely_item(weight_delta, slot._inventory)
        for item_change in item_changes:
            qty = abs(item_change.quantity)
            item_id = item_change.item_id

            # update slot inventory
            current_qty = slot._inventory.get(item_id, 0)
            new_qty = max(0, current_qty - qty)
            if new_qty == 0:
                db.remove_shelf_slot_item(mac_address, slot_id, item_id)
                slot._inventory.pop(item_id, None)
            else:
                db.update_shelf_slot_quantity(mac_address, slot_id, item_id, new_qty)
                slot._inventory[item_id] = new_qty

            # add to cart, needs lock
            with _lock:
                current_cart_qty, _ = CART_DICT.get(item_id, (0, 'shelf'))
                CART_DICT[item_id] = (current_cart_qty + qty, 'shelf')
                CART_ITEM_DATA[item_id] = slot._all_items_by_id[item_id]

            # for ui, if quantity is positive then add to cart
            out = {'id': item_id, 'quantity': qty}
            PI_UART_PORT.write((json.dumps(out) + "\n").encode('utf-8'))
            print(f"{time_str}: Add {qty}x {item_change.name} to cart (slot {slot_id}, shelf {mac_address})")
            print(f"  Cart: {CART_DICT}")

    elif weight_delta > 0:
        # item has been put back, so find based on whats in the cart
        if not CART_DICT:
            return

        cart_items = [CART_ITEM_DATA[iid] for iid in CART_DICT if iid in CART_ITEM_DATA]
        if not cart_items:
            return

        # predict_most_likely_item expects item_id -> quantity; unwrap the tuple values
        cart_quantities = {iid: qty for iid, (qty, _src) in CART_DICT.items()}
        item_changes = slot.predict_most_likely_item(weight_delta, cart_quantities, cart_items)
        for item_change in item_changes:
            qty = abs(item_change.quantity)
            item_id = item_change.item_id

            # remove from cart with lock
            with _lock:
                current_cart_qty, _src = CART_DICT.get(item_id, (0, 'shelf'))
                new_cart_qty = max(0, current_cart_qty - qty)
                if new_cart_qty == 0:
                    CART_DICT.pop(item_id, None)
                    CART_ITEM_DATA.pop(item_id, None)
                else:
                    CART_DICT[item_id] = (new_cart_qty, _src)

            # update slot inventory
            current_qty = slot._inventory.get(item_id, 0)
            new_qty = current_qty + qty
            slot._inventory[item_id] = new_qty
            if current_qty == 0:
                db.add_shelf_slot_item(mac_address, slot_id, item_id, qty)
            else:
                db.update_shelf_slot_quantity(mac_address, slot_id, item_id, new_qty)

            # send with negative quantity to show its been removed
            out = {'id': item_id, 'quantity': -qty}
            PI_UART_PORT.write((json.dumps(out) + "\n").encode('utf-8'))
            print(f"{time_str}: Put back {qty}x {item_change.name} to slot {slot_id} (shelf {mac_address})")
            print(f"  Cart: {CART_DICT}")