import json
from time import sleep
import serial
from typing import List, Tuple
from data_classes import Item
import math
import database as db
import pandas as pd

MAX_ITEM_REMOVALS_TO_CHECK = 3
THRESHOLD_WEIGHT_PROBABILITY = 30

class Slot:

    _previous_weight_value: float
    _items: List[Item]

    def __init__(self, starting_value: float = 0.0):
        self._previous_weight_value = starting_value
        self._items = list()
        self._items = db.get_items()


    def predict_most_likely_item(self, new_weight: float) -> List[Item]:
        """
        Given a change in weight, predict the most likely item that could have been added/removed from the scale.
        :param new_weight: float new weight in grams
        :return: Tuple: Item, Float; The item that is most likely
        """
        weight_delta = new_weight - self._previous_weight_value
        direction = 1 if weight_delta > 0 else -1
        abs_weight_delta = abs(weight_delta)

        # Store probabilities of the various quantities of items being added/removed from the slot
        probabilities = dict()
        # Iterate through all possible items
        for item in self._items:
            # Store probabilities
            probabilities[item.item_id] = []
            # Iterate through all possible quantities
            for potential_quantity in range(1, MAX_ITEM_REMOVALS_TO_CHECK + 1):
                expected_weight = item.avg_weight * potential_quantity
                scaled_std = item.std_weight * (potential_quantity ** 0.5)

                # Calculate log-likelihood using normal distribution
                z_score = (abs_weight_delta - expected_weight) / scaled_std if scaled_std > 0 else float('inf')
                log_likelihood = -0.5 * (z_score ** 2) - math.log(scaled_std)
                # Store this probability
                probabilities[item.item_id].append(log_likelihood)
        # Convert probabilities to pandas dataframe
        df = pd.DataFrame(probabilities, index=range(1, MAX_ITEM_REMOVALS_TO_CHECK + 1)).T
        # Convert to stack to get top n probabilities
        probability_series = df.stack()
        top_n = 1
        top_n_probabilities = probability_series.nlargest(top_n)
        items: List[Item] = list()
        item_ids_and_quantities = dict()

        # Iterate through the top probabilities in decreasing order
        for rank, ((item_id, quantity), probability) in enumerate(top_n_probabilities.items(), start=1):
            print(f'{quantity}x {db.get_item(item_id).name} (p={probability})')
            if abs(probability) > THRESHOLD_WEIGHT_PROBABILITY:
                # If this probability is less than the threshold, all others will be too so return early
                break
            else:
                # Probability is above the threshold, add this item/quantity combination as something that
                # probably happened.
                if item_id in item_ids_and_quantities:
                    # Item id already exists, increase quantity
                    item_ids_and_quantities[item_id] += (direction * quantity)
                else:
                    # Item id does not already exist, add this quantity
                    item_ids_and_quantities[item_id] = (direction * quantity)

            # Create item objects that represent each of the changes
        for item_id in item_ids_and_quantities:
            # Get the quantity for this item
            quantity = item_ids_and_quantities[item_id]
            # Get the existing item object
            existing_item_obj = db.get_item(item_id)
            # Create a new item object with this quantity, and add it to the list of items to be
            # returned
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
            # Return resulting items, where the quantity matches the number predicted to be added/removed from the scale.
        self._previous_weight_value = new_weight
        return items


class Shelf:

    _mac_address: str
    slots: list[Slot]

    def __init__(self, mac_address: str, starting_slot_values: List[float]):
        self._mac_address = mac_address
        self.slots = list()

        for i in range(len(starting_slot_values)):

            self.slots.append(Slot(starting_value=starting_slot_values[i]))


def main():

    mac_address_to_shelves = dict()

    # Set up uart port
    esp_uart_port = serial.Serial(
        port="/dev/ttyTHS1",
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

        # Read weight data from ESP if available
        if esp_uart_port.in_waiting > 0:
            data = esp_uart_port.read(esp_uart_port.in_waiting).decode('utf-8', errors='ignore').strip()
            try:
                json_data = json.loads(data)
            except json.JSONDecodeError:
                print("Unable to decode JSON", data)
                continue

            # Check that json has necessary fields
            if not 'mac_address' in json_data or not 'slot_weights_g' in json_data:
                print("JSON does not have mac_address or slot_weights_g")
                continue

            mac_address = json_data['mac_address']
            if mac_address in mac_address_to_shelves:

                shelf = mac_address_to_shelves[mac_address]

                for i, new_weight in enumerate(json_data['slot_weights_g']):
                    item_changes = shelf.slots[i].predict_most_likely_item(new_weight)
                    for item_change in item_changes:
                        if item_change.quantity > 0:
                            print(f"Remove {item_change.quantity} from cart")
                        else:
                            print(f"Add {item_change.quantity} to cart")

            else:
                # New shelf, just save this as the previous slot data
                shelf = Shelf(mac_address, json_data['slot_weights_g'])
                mac_address_to_shelves[mac_address] = shelf


if __name__ == '__main__':
    main()

