#!/bin/bash

sudo apt update
sudo apt -y full-updrade
sudo apt -y install python3-dev python3-pip git 
sudo apt -y install libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
pip3 install numpy opencv-python  matplotlib pyserial
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
sudo ./scripts/patch-realsense-ubuntu-xenial-joule.sh
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=true
sudo make install
pip3 install pyrealsense2
rm -rf ~/librealsense
