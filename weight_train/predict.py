###############################################################################
#
# File: predict.py
#
# Author: Isaac Ingram
#
# Purpose:
#
###############################################################################
import argparse
import pickle
from model import AdvancedItemWeightDetectionModel
from pathlib import Path
import json
from os import environ
import paho.mqtt.client as mqtt_client
import paho.mqtt.enums as mqtt_enums

# Constants
MQTT_URL = environ.get('MQTT_URL', 'test.mosquitto.org')
MQTT_PORT = environ.get('MQTT_PORT', 1883)
MQTT_DATA_TOPIC = 'shelf/data'

# Global variables
shelf_id = environ.get('SHELF_ID', "")
try:
    slot_id = int(environ.get('SLOT_ID', 0))
except TypeError:
    slot_id = 0
model: AdvancedItemWeightDetectionModel


def on_msg_callback(client, userdata, msg):

    global model

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
                    prediction = model.predict(current_weight)
                    print(prediction)


def main():

    global model

    # Connect to MQTT, subscribe to data topic, and start MQTT thread
    mqtt = mqtt_client.Client(mqtt_enums.CallbackAPIVersion.VERSION2)
    mqtt.connect(MQTT_URL, MQTT_PORT)
    mqtt.subscribe(MQTT_DATA_TOPIC, qos=0)
    mqtt.message_callback_add(MQTT_DATA_TOPIC, on_msg_callback)
    mqtt.loop_start()

    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('model', type=str, help="Path to model file")
    args = arg_parser.parse_args()

    # Load model
    with open(Path.cwd() / Path(args.model), 'rb') as model_file:
        model = pickle.load(model_file)

    while True:
        pass


if __name__ == '__main__':
    main()
