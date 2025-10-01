###############################################################################
#
# File: store_images.py
#
# Author: Isaac Ingram
#
# Purpose: Store an image stream from a website to the local disk.
#
# Usage: This program requires the requests library to be installed.
# To use, provide two commandline arguments: <URL> and <path>. 'URL' is the URL
# to requests content from. 'path' is the relative or absolute path to where
# the images should be placed on your system.
#
###############################################################################
import sys
from datetime import datetime

import requests


def print_usage_and_exit() -> None:
    """
    Print the usage statement and exit.
    :return: None
    """
    print("Usage: store_images.py <URL> <path>")
    exit(0)


def main():
    try:
        while(True):
            # Check command line args
            if len(sys.argv) != 3:
                print_usage_and_exit()
            url = sys.argv[1]
            storage_path = sys.argv[2]
            # Get rid of last character in path if it's a slash
            if storage_path[-1] == "/":
                storage_path = storage_path[:-1]
            try:
                image_data = requests.get(url).content
                img_name = f'{storage_path}/{datetime.now().strftime("%S-%M-%H-%d-%m-%Y.jpg")}'
                with open(img_name, 'wb') as image_file:
                    image_file.write(image_data)
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")


if __name__ == '__main__':
    main()