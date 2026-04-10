import json
from time import sleep
import serial
from typing import List, Tuple, Dict
import threading
import websockets

from data_classes import Item
import math
import database as db
import pandas as pd
import time
from pathlib import Path
import asyncio
import websockets

MAX_ITEM_REMOVALS_TO_CHECK = 3
THRESHOLD_WEIGHT_PROBABILITY = 30

ESP_SERIAL_PORT = "/dev/ttyTHS1"
PI_SERIAL_PORT = "/dev/ttyUSB0"

message_queue = asyncio.Queue()

async def ui_handler(websocket):
    while True:
        message = await message_queue.get()
        await websocket.send(message)

async def websocket_server():
    async with websockets.serve(ui_handler, "localhost", 8765):
        await asyncio.Future()

class Slot:

    _items: List[Item]
    _items_by_id: Dict[int, Item]

    def __init__(self, items: List[Item] = None):
        self._items = list()

        if items is not None:
            for item in items:
                self._items.append(item)

        self._items_by_id = dict()
        for item in self._items:
            self._items_by_id[item.item_id] = item


    def predict_most_likely_item(self, weight_delta: float) -> List[Item]:
        direction = 1 if weight_delta > 0 else -1
        abs_weight_delta = abs(weight_delta)

        probabilities = dict()
        for item in self._items:
            probabilities[item.item_id] = []
            for potential_quantity in range(1, MAX_ITEM_REMOVALS_TO_CHECK + 1):
                expected_weight = item.avg_weight * potential_quantity
                scaled_std = item.std_weight * (potential_quantity ** 0.5)

                z_score = (abs_weight_delta - expected_weight) / scaled_std if scaled_std > 0 else float('inf')
                log_likelihood = -0.5 * (z_score ** 2) - math.log(scaled_std)
                probabilities[item.item_id].append(log_likelihood)

        df = pd.DataFrame(probabilities, index=range(1, MAX_ITEM_REMOVALS_TO_CHECK + 1)).T
        probability_series = df.stack()
        top_n = 1
        top_n_probabilities = probability_series.nlargest(top_n)
        items: List[Item] = list()
        item_ids_and_quantities = dict()

        for rank, ((item_id, quantity), probability) in enumerate(top_n_probabilities.items(), start=1):
            if abs(probability) > THRESHOLD_WEIGHT_PROBABILITY:
                break
            else:
                if item_id in item_ids_and_quantities:
                    item_ids_and_quantities[item_id] += (direction * quantity)
                else:
                    item_ids_and_quantities[item_id] = (direction * quantity)

        for item_id in item_ids_and_quantities:
            quantity = item_ids_and_quantities[item_id]
            existing_item_obj = self._items_by_id[item_id]
            items.append(
                Item(
                    item_id,
                    existing_item_obj.name,
                    existing_item_obj.upc,
                    existing_item_obj.price,
                    quantity,
                    existing_item_obj.avg_weight,
                    existing_item_obj.std_weight,
                    existing_item_obj.thumbnail_url,
                    existing_item_obj.vision_class
                )
            )
        return items


class Shelf:

    _mac_address: str
    slots: list[Slot]

    def __init__(self, mac_address: str, slots: List[Slot] = None):
        self._mac_address = mac_address
        self.slots = list()

        if slots is not None:
            for slot in slots:
                self.slots.append(slot)


def load_stock_file(file_path: Path = Path("stock.json")):

    with open(file_path) as file:
        json_data = json.load(file)

    shelves = list()
    for shelf_mac in json_data:
        slots = list()
        for slot_i, json_slot in enumerate(json_data[shelf_mac]):

            items = list()
            for json_item in json_slot:
                item = db.get_item(item_id=json_item["id"])
                item.quantity = json_item["quantity"]
                print("Loaded data from stock store")
                items.append(item)

            slot = Slot(items=items)
            slots.append(slot)

        shelf = Shelf(shelf_mac, slots=slots)
        shelves.append(shelf)
    return shelves


def main():

    loop = asyncio.new_event_loop()

    def run_async():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_server())

    t = threading.Thread(target=run_async, daemon=True)
    t.start()

    mac_address_to_shelves = dict()
    known_shelves = load_stock_file()
    for shelf in known_shelves:
        mac_address_to_shelves[shelf._mac_address] = shelf

    esp_uart_port = serial.Serial(
        port=ESP_SERIAL_PORT,
        baudrate=115200,
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
            json_data = json.loads(line)
        except json.JSONDecodeError:
            print("Unable to decode JSON", line)
            continue

        if not 'shelf_mac' in json_data or not 'slot_id' in json_data or 'delta_g' not in json_data:
            print("JSON does not have shelf_mac, or slot_id, or delta_g")
            continue

        mac_address = json_data['shelf_mac']
        if mac_address in mac_address_to_shelves:

            shelf = mac_address_to_shelves[mac_address]
            slot = shelf.slots[json_data['slot_id']]

            item_changes = slot.predict_most_likely_item(json_data['delta_g'])

            for item_change in item_changes:
                out = {
                    'id': item_change.item_id,
                    'quantity': -item_change.quantity
                }
                json_str = json.dumps(out)

                # Send to UI
                asyncio.run_coroutine_threadsafe(message_queue.put(json_str), loop)

                time_str = time.strftime("%H:%M:%S.") + f"{int((time.time() * 1000) % 1000):03d}"
                if item_change.quantity > 0:
                    print(f"{time_str}: Remove {abs(item_change.quantity)} {item_change.name} from cart")
                else:
                    print(f"{time_str}: Add {abs(item_change.quantity)} {item_change.name} to cart")

        else:
            print("Unknown shelf joined")
            shelf = Shelf(mac_address)
            mac_address_to_shelves[mac_address] = shelf


if __name__ == '__main__':
    main()