# Bits 'n Bytes (AI)
Bits 'n Bytes is a next generation vending machine by Computer Science House. This repository contains the software for our AI pipeline that uses [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)in conjunction with a custom detection algorithm and training dataset. YOLO is meant for autonomous vehicles, but our training set focuses on consumer items such as Sour Patch Kids, Brownie Brittle, and Little Bites.
# Local Setup
To contribute to this repository, you will first need to set up a local running version.
## 1. Clone this repository
Clone this repository somewhere on your system.
## 2. Install Git submodules
In order to run any of the software, we rely on you also having [yolov7](https://github.com/WongKinYiu/yolov7) installed within the same directory. The easiest and least error-prone way to do this is navigate into your checkout for this repository and run:
```
git submodule update --init --recursive
```
