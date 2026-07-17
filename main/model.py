import serial
from typing import List, Dict, Optional
import math
from config import *

class Item:

    def __init__(
            self, item_id, name, upc, price, quantity, avg_weight, std_weight,
            thumbnail_url, vision_class
    ):
        self.item_id = item_id
        self.name = name
        self.upc = upc
        self.price = price
        self.quantity = quantity
        self.avg_weight = avg_weight
        self.std_weight = std_weight
        self.thumbnail_url = thumbnail_url
        self.vision_class = vision_class

    def __str__(self):
        return (f'Item[{self.item_id},{self.name},UPC:{self.upc},${self.price},'
                f'{self.units}units,{self.avg_weight}{WEIGHT_UNIT},'
                f'{self.std_weight}{WEIGHT_UNIT},{self.thumbnail_url},'
                f'{self.vision_class}]')

    def __eq__(self, other):
        if isinstance(other, Item):
            return self.item_id == other.item_id
        else:
            return False

    def __hash__(self):
        return hash(self.item_id)

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
            if sid not in self.slots:
                slot = Slot(self._mac_address, sid, slot_items)
                slot.set_inventory({r['id']: r['quantity'] for r in slot_data.get('items', [])})
                self.slots[sid] = slot
            else:
                existing = self.slots[sid]
                existing._all_items = slot_items
                existing._all_items_by_id = {item.item_id: item for item in slot_items}
                existing._inventory = {r['id']: r['quantity'] for r in slot_data.get('items', [])}
        print(f"Shelf {self._mac_address}: synced {len(self.slots)} slots from DB")