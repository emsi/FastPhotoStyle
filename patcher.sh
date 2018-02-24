#!/bin/sh 

wget  --no-check-certificate https://github.com/NVIDIA/pynvrtc/commit/6417a2896ff8a99f2c4d4195de657671a77c89a0.diff
cwd=$(pwd)
cd /anaconda3/lib/python3.6/site-packages/pynvrtc/ 
patch < ${cwd}/6417a2896ff8a99f2c4d4195de657671a77c89a0.diff 
cd ${cwd}
