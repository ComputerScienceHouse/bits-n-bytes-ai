from model import Slot, Shelf, Item
import json
from time import sleep
import serial
from typing import Dict
import database as db
import time
from config import *

PI_UART_PORT = None
ESP_UART_PORT = None
CLEAR_CART_CMD = bytes([0xDE, 0xAD, 0xBE, 0xEF])

def cart_service_init():
    global PI_UART_PORT
    global ESP_UART_PORT

    mac_address_to_shelves: Dict[str, Shelf] = {}
    cart: Dict[int, int] = {}  # item_id -> quantity currently in cart
    cart_item_data: Dict[int, Item] = {}  # item_id -> Item object (survives DB reloads)

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

def cart_service_loop():
    global CLEAR_CART_CMD
    
    # Check for a clear-cart command from the UI before processing shelf data
    if PI_UART_PORT.in_waiting >= len(CLEAR_CART_CMD):
        incoming = PI_UART_PORT.read(len(CLEAR_CART_CMD))
        if incoming == CLEAR_CART_CMD:
            cart.clear()
            cart_item_data.clear()
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

    if mac_address not in mac_address_to_shelves:
        mac_address_to_shelves[mac_address] = Shelf(mac_address)

    shelf = mac_address_to_shelves[mac_address]
    shelf._load_from_db()
    slot = shelf.get_slot(slot_id)
    time_str = time.strftime("%H:%M:%S.") + f"{int((time.time() * 1000) % 1000):03d}"

    if weight_delta < 0:
        # Item removed from shelf — predict from all items
        item_changes = slot.predict_most_likely_item(weight_delta, slot._inventory)
        for item_change in item_changes:
            qty = abs(item_change.quantity)
            item_id = item_change.item_id

            # Update slot inventory
            current_qty = slot._inventory.get(item_id, 0)
            new_qty = max(0, current_qty - qty)
            if new_qty == 0:
                db.remove_shelf_slot_item(mac_address, slot_id, item_id)
                slot._inventory.pop(item_id, None)
            else:
                db.update_shelf_slot_quantity(mac_address, slot_id, item_id, new_qty)
                slot._inventory[item_id] = new_qty

            # Add to cart
            cart[item_id] = cart.get(item_id, 0) + qty
            cart_item_data[item_id] = slot._all_items_by_id[item_id]

            # UI: positive quantity = add to cart
            out = {'id': item_id, 'quantity': qty}
            PI_UART_PORT.write((json.dumps(out) + "\n").encode('utf-8'))
            print(f"{time_str}: Add {qty}x {item_change.name} to cart (slot {slot_id}, shelf {mac_address})")
            print(f"  Cart: {cart}")

    elif weight_delta > 0:
        # Item put back — predict only from cart items
        if not cart:
            return

        cart_items = [cart_item_data[iid] for iid in cart if iid in cart_item_data]
        if not cart_items:
            return

        item_changes = slot.predict_most_likely_item(weight_delta, cart, cart_items)
        for item_change in item_changes:
            qty = abs(item_change.quantity)
            item_id = item_change.item_id

            # Remove from cart
            new_cart_qty = max(0, cart.get(item_id, 0) - qty)
            if new_cart_qty == 0:
                cart.pop(item_id, None)
                cart_item_data.pop(item_id, None)
            else:
                cart[item_id] = new_cart_qty

            # Update slot inventory
            current_qty = slot._inventory.get(item_id, 0)
            new_qty = current_qty + qty
            slot._inventory[item_id] = new_qty
            if current_qty == 0:
                db.add_shelf_slot_item(mac_address, slot_id, item_id, qty)
            else:
                db.update_shelf_slot_quantity(mac_address, slot_id, item_id, new_qty)

            # UI: negative quantity = remove from cart (put back)
            out = {'id': item_id, 'quantity': -qty}
            PI_UART_PORT.write((json.dumps(out) + "\n").encode('utf-8'))
            print(f"{time_str}: Put back {qty}x {item_change.name} to slot {slot_id} (shelf {mac_address})")
            print(f"  Cart: {cart}")

def main():
    global PI_UART_PORT
    global ESP_UART_PORT

    cart_service_init()

    while True:
        sleep(0.1)
        cart_service_loop()

        


if __name__ == '__main__':
    main()