#!/bin/sh
export DEBUG=1
export RTSP_STREAM="IMG_0332.MOV"
# export RTSP_STREAM="rtsp://192.168.10.109:8554/live.sdp"
export MQTT_BROKER="localhost"
# export MQTT_HOST="host.docker.internal"
export MQTT_PORT=1883
export MQTT_TOPIC="yolo"
export TRACKING_AREAS='[{"tag": "test", "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]}]'