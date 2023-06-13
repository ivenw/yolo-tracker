import json
from dataclasses import dataclass
import cv2

import numpy as np

from yolo_demo.mqtt import MqttClient
from yolo_demo.tracking import DetectedObject, TrackingArea


@dataclass
class Publisher:
    mqtt_client: MqttClient
    root_topic: str

    def publish_event_message(
        self,
        area: TrackingArea,
        object: DetectedObject,
        unix_timestamp_sec: int,
        in_area: bool,
    ) -> None:
        message = {
            "object_id": object.id,
            "in_area": in_area,
            "unix_timestamp_sec": unix_timestamp_sec,
            "class_id": object.class_id,
            "class_name": object.class_name,
            "detection_confidence": object.detection_confidence,
        }
        self.mqtt_client.publish(
            f"{self.root_topic}/{area.tag}/events",
            json.dumps(message, indent=2),
        )

    def publish_count_message(
        self,
        area: TrackingArea,
        object_count: int,
        unix_timestamp_sec: int,
    ) -> None:
        message = {
            "object_count": object_count,
            "unix_timestamp_sec": unix_timestamp_sec,
        }
        self.mqtt_client.publish(
            f"{self.root_topic}/{area.tag}/count",
            json.dumps(message, indent=2),
        )

    def publish_snapshot(
        self,
        image: np.ndarray,
        unix_timestamp_sec: int,
    ) -> None:
        image_jpeg = cv2.imencode(".jpg", image)[1].tobytes()

        self.mqtt_client.publish(f"{self.root_topic}/snapshot", image_jpeg)
