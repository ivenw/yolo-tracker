import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Union, cast

import cv2
import numpy as np
from numpy import ndarray
from shapely.geometry import Point, Polygon
from torch import Tensor
from torch.cuda import is_available as cuda_is_available
from typing_extensions import Self
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes, Masks, Results

from yolo_demo.annotation import FrameAnnotator
from yolo_demo.mqtt import DummyMqttClient, MqttClient, PahoMqttClient
from yolo_demo.tracking import DetectedObject, TrackingArea

DEBUG = True
# DEBUG_RTSP_STREAM = "rtsp://192.168.10.109:8554/live.sdp"
DEBUG_RTSP_STREAM = "IMG_0327.jpg"
DEBUG_MQTT_HOST = "localhost"
DEBUG_MQTT_PORT = 1883
DEBUG_MQTT_TOPIC = "yolo"
DEBUG_DETECTION_AREA_TAG = "test"
DEBUG_DETECTION_AREA_CONFIDENCE_THRESHOLD = 0.5
DEBUG_DETECTION_AREA_POLYGON = Polygon(
    [
        (0.27, 0.76),
        (0.77, 0.67),
        (1.0, 0.88),
        (1.0, 1.0),
        (0.34, 1.0),
    ]
)
# DEBUG_DETECTION_AREA_POLYGON = Polygon(
#     [
#         (0.5, 0.0),
#         (1.0, 0.0),
#         (1.0, 1.0),
#         (0.5, 1.0),
#     ]
# )


@dataclass
class AppConfig:
    rtsp_stream: str
    mqtt_broker: str
    mqtt_port: int
    mqtt_topic: str
    tracking_areas: list[TrackingArea]
    mqtt_user: Optional[str]
    mqtt_password: Optional[str]

    @classmethod
    def from_env(cls) -> Self:
        rtsp_stream = os.getenv("RTSP_STREAM")
        mqtt_broker = os.getenv("MQTT_BROKER")
        mqtt_port = os.getenv("MQTT_PORT")
        mqtt_topic = os.getenv("MQTT_TOPIC")
        detection_areas = os.getenv("TRACKING_AREAS")

        if not rtsp_stream:
            raise ValueError("Environment variable 'RTSP_STREAM' is not set")
        if not mqtt_broker:
            raise ValueError("Environment variable 'MQTT_HOST' is not set")
        if not mqtt_port:
            raise ValueError("Environment variable 'MQTT_PORT' is not set")
        if not mqtt_topic:
            raise ValueError("Environment variable 'MQTT_TOPIC' is not set")
        if not detection_areas:
            raise ValueError("Environment variable 'TRACKING_AREAS' is not set")

        return cls(
            rtsp_stream=rtsp_stream,
            mqtt_broker=mqtt_broker,
            mqtt_port=int(mqtt_port),
            mqtt_topic=mqtt_topic,
            tracking_areas=detection_areas_from_json(detection_areas),
            mqtt_user=os.getenv("MQTT_USER"),
            mqtt_password=os.getenv("MQTT_PASSWORD"),
        )


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


def detect_and_track(rtsp_stream: str) -> Iterator[Results]:
    """Detect and track objects in a video stream."""
    # TODO: make this more fault tolerant when the stream is not available. Don't crash
    model = YOLO("yolov8n-seg.pt")
    if cuda_is_available() is True:
        return model.track(rtsp_stream, stream=True, verbose=False, classes=0, device=0)
    return model.track(rtsp_stream, stream=True, verbose=False, classes=0)


def filter_valid_objects_from_results(results: Results) -> Iterator[DetectedObject]:
    if results.boxes and results.masks:
        for box, mask in zip(results.boxes, results.masks):  # type: ignore fallback to '__get_item__'
            object = DetectedObject.from_box_mask_result(box, mask, results)
            if object.id is None:
                continue
            yield object


def area_contains_object(
    detection_area: TrackingArea, detected_object: DetectedObject
) -> bool:
    """Check if a detection area contains a detected object.

    Assumes that detection area is in plane of an even floor and that the object is
    standing on the floor.
    """
    return detection_area.polygon.contains(detected_object.max_segment_y_point)


def detection_areas_from_json(s: str, /) -> list[TrackingArea]:
    data = json.loads(s)
    return [
        TrackingArea(
            tag=d["tag"],
            polygon=Polygon(d["area"]),
        )
        for d in data
    ]


def analyze_results_and_publish(
    results_stream: Iterable[Results],
    tracking_areas: list[TrackingArea],
    publisher: Publisher,
) -> None:
    object_record: dict[str, set[DetectedObject]] = {
        k.tag: set() for k in tracking_areas
    }

    for results in results_stream:
        unix_timestamp_sec = int(time.time())
        this_frame_detected_objects: list[DetectedObject] = []

        for area in tracking_areas:
            previous_frame_object_count = len(object_record[area.tag])
            per_area_frame_object_record: set[DetectedObject] = set()

            for object in filter_valid_objects_from_results(results):
                this_frame_detected_objects.append(object)
                if area_contains_object(area, object) is False:
                    continue
                per_area_frame_object_record.add(object)

                if object not in object_record[area.tag]:
                    object_record[area.tag].add(object)
                    publisher.publish_event_message(
                        area,
                        object,
                        unix_timestamp_sec,
                        in_area=True,
                    )

            no_longer_in_area = object_record[area.tag].difference(
                per_area_frame_object_record
            )

            object_record[area.tag].intersection_update(per_area_frame_object_record)
            for object in no_longer_in_area:
                publisher.publish_event_message(
                    area,
                    object,
                    unix_timestamp_sec,
                    in_area=False,
                )

            this_frame_object_count = len(per_area_frame_object_record)
            if this_frame_object_count != previous_frame_object_count:
                publisher.publish_count_message(
                    area,
                    this_frame_object_count,
                    unix_timestamp_sec,
                )

        if DEBUG:
            FrameAnnotator(results).annotate_tracking_areas(
                tracking_areas
            ).annotate_object_tracking_position(this_frame_detected_objects).show()


def main() -> None:
    if DEBUG:
        app_config = AppConfig(
            rtsp_stream=DEBUG_RTSP_STREAM,
            mqtt_broker=DEBUG_MQTT_HOST,
            mqtt_port=DEBUG_MQTT_PORT,
            mqtt_topic=DEBUG_MQTT_TOPIC,
            tracking_areas=[
                TrackingArea(DEBUG_DETECTION_AREA_TAG, DEBUG_DETECTION_AREA_POLYGON),
                TrackingArea(
                    "test2", Polygon([(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)])
                ),
            ],
            mqtt_user=None,
            mqtt_password=None,
        )
    else:
        app_config = AppConfig.from_env()

    if DEBUG:
        mqtt_client = DummyMqttClient()
    else:
        mqtt_client = PahoMqttClient()
        mqtt_client.connect(
            app_config.mqtt_broker,
            app_config.mqtt_port,
            app_config.mqtt_user,
            app_config.mqtt_password,
        )

    publisher = Publisher(mqtt_client, app_config.mqtt_topic)
    results = detect_and_track(app_config.rtsp_stream)

    analyze_results_and_publish(results, app_config.tracking_areas, publisher)

    if DEBUG:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
