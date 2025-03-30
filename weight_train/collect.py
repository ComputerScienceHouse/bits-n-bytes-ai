###############################################################################
#
# File: collect.py
#
# Author: Isaac Ingram
#
# Purpose:
#
###############################################################################
import datetime
import os
import time
from os import environ
import sys
from pathlib import Path
import paho.mqtt.client as mqtt_client
import json
import paho.mqtt.enums as mqtt_enums
import csv


# Constants
MQTT_URL = environ.get('MQTT_URL', 'test.mosquitto.org')
MQTT_PORT = environ.get('MQTT_PORT', 1883)
MQTT_DATA_TOPIC = 'shelf/data'
TMP_MQTT_DATA_PATH = Path.cwd() / Path('tmp/mqtt-data.csv')
TMP_INTERACTION_DATA_PATH = Path.cwd() / Path('tmp/interaction-data.csv')
TIME_STR_FT = '%H:%M:%S:%f-%d-%m-%Y'

# Global variables
shelf_id = environ.get('SHELF_ID', "")
try:
    slot_id = int(environ.get('SLOT_ID', 0))
except TypeError:
    slot_id = 0
window_ms = 5000
known_weight_g = environ.get('KNOWN_WEIGHT', 226)


def print_help():
    print("exit - Exit this app.")
    print("set shelf <mac address> - configure which shelf to listen to for data.")
    print("set slot <slot id> - configure which slot id to listen to for data.")
    print("set window <millis> - set data collection window in millis")
    print("set weight <grams> - set known weight being added/removed from scale")
    print("record <integer> - record weight data for 5 seconds, during which you will add/remove a certain number of items that you enter as the integer. Positive numbers for items being removed, negative for items being put back.")


def on_msg_callback(client, userdata, msg):
    """
    Callback for when a MQTT message is received on the shelf data topic.
    Writes time series data to a CSV file along with the weights.
    :param client:
    :param userdata:
    :param msg:
    :return:
    """

    global shelf_id
    global window_ms
    global slot_id
    # Get the time that this message was received
    current_time = datetime.datetime.now().strftime(TIME_STR_FT)
    # Load message as JSON
    message = msg.payload.decode('utf-8')
    json_data = json.loads(message)
    # Check that message contains all necessary fields
    # Message is from the correct shelf
    if 'id' in json_data:
        if json_data['id'] == shelf_id:
            # Message contains 'data' field
            if 'data' in json_data:
                # Data field is at least the length of the target slot ID
                if len(json_data['data']) > slot_id:
                    # Get the current weight
                    current_weight = json_data['data'][slot_id]
                    # Write weight to CSV file
                    with open(TMP_MQTT_DATA_PATH, 'a') as out_file:
                        csv_writer = csv.writer(out_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow([current_time, current_weight])


def main():

    global shelf_id
    global window_ms
    global slot_id
    global known_weight_g

    # Connect to MQTT, subscribe to data topic, and start MQTT thread
    mqtt = mqtt_client.Client(mqtt_enums.CallbackAPIVersion.VERSION2)
    mqtt.connect(MQTT_URL, MQTT_PORT)
    mqtt.subscribe(MQTT_DATA_TOPIC, qos=0)
    mqtt.message_callback_add(MQTT_DATA_TOPIC, on_msg_callback)
    mqtt.loop_start()


    # Create mqtt data csv if it doesn't exist
    if not TMP_MQTT_DATA_PATH.exists():
        os.makedirs(os.path.dirname(TMP_MQTT_DATA_PATH), exist_ok=True)
        with open(TMP_MQTT_DATA_PATH, 'w') as file:
            pass

    # Create interaction data csv if it doesn't exist
    if not TMP_INTERACTION_DATA_PATH.exists():
        os.makedirs(os.path.dirname(TMP_INTERACTION_DATA_PATH), exist_ok=True)
        with open(TMP_INTERACTION_DATA_PATH, 'w') as file:
            pass

    # Print user instructions
    print("Welcome to weight_train collect. Type 'help' for a list of commands or 'exit' to exit.")

    # Main CLI loop
    while True:
        command = input('>').split()
        match command:
            case ['help']:
                print_help()
            case ['exit']:
                # TODO parse data
                break
            case ['set', 'shelf', *arguments]:
                shelf_id = arguments[0]
            case ['set', 'slot', *arguments]:
                slot_id = int(arguments[0])
            case ['set', 'weight', *arguments]:
                known_weight_g = int(arguments[0])
            case ['set', 'window', *arguments]:
                try:
                    window_ms = int(arguments[0])
                except ValueError or IndexError:
                    print("Invalid arguments")
            case ['record', *arguments]:
                current_time = datetime.datetime.now().strftime(TIME_STR_FT)
                if len(arguments) != 1:
                    print("Unknown number of args")
                else:
                    # Make sure number entered is valid
                    try:
                        num_items = int(arguments[0])
                    except TypeError:
                        print("Invalid number")
                        continue
                    # Write instruction message to terminal
                    if num_items > 0:
                        print("Remove %d items from slot %d..." % (num_items, slot_id))
                    elif num_items < 0:
                        print("Add %d items to slot %d..." % (-1 * num_items, slot_id))
                    else:
                        print("Do not touch items...")
                    # Pause however many seconds
                    start_time_ms = time.time() * 1000
                    while time.time() * 1000 < start_time_ms + window_ms:
                        pass
                    # Write output to terminal
                    with open(TMP_INTERACTION_DATA_PATH, 'a') as out_file:
                        csv_writer = csv.writer(out_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow([current_time, known_weight_g, num_items])
                    print("Recorded. If you did not complete instruction, type 'delete' to remove the last datapoint.")
            case ['delete']:
                with open(TMP_INTERACTION_DATA_PATH, 'r') as in_file:
                    csv_reader = csv.reader(in_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    rows = list()
                    for row in csv_reader:
                        rows.append(row)
                with open(TMP_INTERACTION_DATA_PATH, 'w') as out_file:
                    csv_writer = csv.writer(out_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerows(rows[:-1])
                print("Deleted last datapoint.")
            case _:
                print("Unknown command, type 'help' for a list of commands.")


if __name__ == '__main__':
    main()
