import json
from time import sleep
import serial
from typing import List, Dict, Optional
from data_classes import Item
import math
import database as db
import time

THRESHOLD_WEIGHT_PROBABILITY = 30

ESP_SERIAL_PORT = "/dev/ttyUSB0"
PI_SERIAL_PORT = "/dev/ttyTHS1"


class Slot:

    _all_items: List[Item]
    _all_items_by_id: Dict[int, Item]
    _inventory: Dict[int, int]  # item_id -> quantity currently in this slot

    def __init__(self, shelf_id: str, slot_id: int, slot_items: Optional[List[Item]] = None):
        self._shelf_id = shelf_id
        self._slot_id = slot_id
        self._inventory = {}
        self._all_items = slot_items if slot_items is not None else list()
        self._all_items_by_id = {item.item_id: item for item in self._all_items}

    def set_inventory(self, inventory: Dict[int, int]):
        self._inventory = inventory

    def predict_most_likely_item(self, weight_delta: float, max_quantities: Dict[int, int], candidate_items: Optional[List[Item]] = None) -> List[Item]:
        """
        Given a change in weight, predict the most likely item that could have been added/removed.
        :param weight_delta: change in weight in grams
        :param max_quantities: item_id -> max quantity to check (slot inventory or cart quantities)
        :param candidate_items: items to consider; defaults to all items if None
        :return: list of Items with quantity indicating how many were added (positive) or removed (negative)
        """
        if candidate_items is None:
            candidate_items = self._all_items

        if not candidate_items:
            return []

        direction = 1 if weight_delta > 0 else -1
        abs_weight_delta = abs(weight_delta)

        all_scores: List[tuple] = []  # (log_likelihood, item_id, quantity)
        for item in candidate_items:
            max_qty = max_quantities.get(item.item_id, 1)
            for potential_quantity in range(1, max_qty + 1):
                expected_weight = item.avg_weight * potential_quantity
                scaled_std = item.std_weight * (potential_quantity ** 0.5)
                z_score = (abs_weight_delta - expected_weight) / scaled_std if scaled_std > 0 else float('inf')
                log_likelihood = -0.5 * (z_score ** 2) - math.log(scaled_std)
                all_scores.append((log_likelihood, item.item_id, potential_quantity))

        all_scores.sort(key=lambda x: x[0], reverse=True)
        top_n_probabilities = all_scores[:1]
        item_ids_and_quantities: Dict[int, int] = {}

        for probability, item_id, quantity in top_n_probabilities:
            if abs(probability) > THRESHOLD_WEIGHT_PROBABILITY:
                break
            if item_id in item_ids_and_quantities:
                item_ids_and_quantities[item_id] += direction * quantity
            else:
                item_ids_and_quantities[item_id] = direction * quantity

        candidate_by_id = {item.item_id: item for item in candidate_items}

        items: List[Item] = []
        for item_id, quantity in item_ids_and_quantities.items():
            existing = candidate_by_id.get(item_id) or self._all_items_by_id.get(item_id)
            if existing is None:
                continue
            items.append(Item(
                item_id,
                existing.name,
                existing.upc,
                existing.price,
                quantity,
                existing.avg_weight,
                existing.std_weight,
                existing.thumbnail_url,
                existing.vision_class
            ))
        return items


class Shelf:

    _mac_address: str
    slots: Dict[int, Slot]

    def __init__(self, mac_address: str):
        self._mac_address = mac_address
        self.slots: Dict[int, Slot] = {}

        self._load_from_db()

    def get_slot(self, slot_id: int) -> Slot:
        if slot_id not in self.slots:
            self._load_from_db()
        if slot_id not in self.slots:
            self.slots[slot_id] = Slot(self._mac_address, slot_id)
        return self.slots[slot_id]

    def _load_from_db(self):
        shelf_data = db.get_shelf_contents(self._mac_address)
        for slot_data in shelf_data:
            sid = slot_data['slot_id']
            slot_items = list()
            for r in slot_data.get('items', []):
                slot_items.append(Item(
                    r['id'], r['name'], r['upc'], r['price'], r['quantity'],
                    r['weight_avg'], r['weight_std'], r['thumb_img'], r['vision_class']
                ))
                print(f"Added new item {r['name']}")
            if sid not in self.slots:
                print("ADDING BRAND NEW SLOT")
                slot = Slot(self._mac_address, sid, slot_items)
                slot.set_inventory({r['id']: r['quantity'] for r in slot_data.get('items', [])})
                self.slots[sid] = slot
            else:
                existing = self.slots[sid]
                existing._all_items = slot_items
                existing._all_items_by_id = {item.item_id: item for item in slot_items}
                existing._inventory = {r['id']: r['quantity'] for r in slot_data.get('items', [])}
        print(f"Shelf {self._mac_address}: synced {len(self.slots)} slots from DB")


def main():

    mac_address_to_shelves: Dict[str, Shelf] = {}
    cart: Dict[int, int] = {}  # item_id -> quantity currently in cart
    cart_item_data: Dict[int, Item] = {}  # item_id -> Item object (survives DB reloads)

    esp_uart_port = serial.Serial(
        port=ESP_SERIAL_PORT,
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )
    pi_uart_port = serial.Serial(
        port=PI_SERIAL_PORT,
        baudrate=9600,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )

    while True:

        sleep(0.1)

        line = esp_uart_port.readline().decode('utf-8', errors='ignore').strip()

        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print("Unable to decode JSON", line)
            continue

        if 'shelf_mac' not in msg or 'slot_id' not in msg or 'delta_g' not in msg:
            print("JSON missing shelf_mac, slot_id, or delta_g")
            continue

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
                pi_uart_port.write((json.dumps(out) + "\n").encode('utf-8'))
                print(f"{time_str}: Add {qty}x {item_change.name} to cart (slot {slot_id}, shelf {mac_address})")
                print(f"  Cart: {cart}")

        elif weight_delta > 0:
            # Item put back — predict only from cart items
            if not cart:
                continue

            cart_items = [cart_item_data[iid] for iid in cart if iid in cart_item_data]
            if not cart_items:
                continue

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
                pi_uart_port.write((json.dumps(out) + "\n").encode('utf-8'))
                print(f"{time_str}: Put back {qty}x {item_change.name} to slot {slot_id} (shelf {mac_address})")
                print(f"  Cart: {cart}")


if __name__ == '__main__':
    main()
