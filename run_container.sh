#!/bin/sh
docker run -it -e DEBUG -e RTSP_STREAM -e MQTT_BROKER -e MQTT_PORT -e MQTT_TOPIC -e TRACKING_AREAS yolo-demo