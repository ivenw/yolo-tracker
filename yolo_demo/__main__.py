from itertools import product
from typing_extensions import Self

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Union, cast
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results, Boxes, Masks
from torch import Tensor
from numpy import ndarray
import numpy as np

import time


import os
import json

from shapely.geometry import Polygon, Point

from yolo_demo.mqtt import DummyMqttClient, MqttClient, PahoMqttClient


DEBUG = True
DEBUG_RTSP_STREAM = "rtsp://192.168.10.109:8554/live.sdp"
# DEBUG_RTSP_STREAM = "IMG_0327.jpg"
DEBUG_MQTT_HOST = "localhost"
# DEBUG_MQTT_HOST = "host.docker.internal"
DEBUG_MQTT_PORT = 1883
DEBUG_MQTT_TOPIC = "yolo"
DEBUG_DETECTION_AREA_TAG = "test"
DEBUG_DETECTION_AREA_CONFIDENCE_THRESHOLD = 0.5
# DEBUG_DETECTION_AREA_POLYGON = Polygon(
#     [
#         (0.27, 0.77),
#         (0.27, 1),
#         (1, 1),
#         (1, 0.77),
#     ]
# )
DEBUG_DETECTION_AREA_POLYGON = Polygon(
    [
        (0.5, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.5, 1.0),
    ]
)


@dataclass
class TrackingArea:
    tag: str
    polygon: Polygon


@dataclass
class DetectedObject:
    id: Optional[int]
    class_id: int
    class_name: str
    detection_confidence: float
    bounding_box_normalized: Union[Tensor, ndarray]
    segment_normalized: ndarray

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DetectedObject):
            return False
        return self.id == o.id

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_box_mask_result(cls, box: Boxes, mask: Masks, result: Results) -> Self:
        class_id = int(cast(float, box.cls[0].tolist()))
        return cls(
            id=int(cast(float, box.id[0].tolist())) if box.id else None,
            class_id=class_id,
            class_name=result.names[class_id],
            detection_confidence=cast(float, box.conf[0].tolist()),
            bounding_box_normalized=box.xyxyn[0],
            segment_normalized=mask.xyn[0],
        )

    @property
    def max_segment_y_point(self) -> Point:
        point_coords_idx = np.where(
            self.segment_normalized == np.max(self.segment_normalized[:, 1])
        )[0]
        point_coords = self.segment_normalized[point_coords_idx]
        return Point(point_coords[0][0], point_coords[0][1])


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


@dataclass
class AppConfig:
    rtsp_stream: str
    mqtt_host: str
    mqtt_port: int
    mqtt_topic: str
    tracking_areas: list[TrackingArea]

    @classmethod
    def from_env(cls) -> Self:
        rtsp_stream = os.getenv("RTSP_STREAM")
        mqtt_host = os.getenv("MQTT_HOST")
        mqtt_port = os.getenv("MQTT_PORT")
        mqtt_topic = os.getenv("MQTT_TOPIC")
        detection_areas = os.getenv("DETECTION_AREAS")

        if not rtsp_stream:
            raise ValueError("Environment variable 'RTSP_STREAM' is not set")
        if not mqtt_host:
            raise ValueError("Environment variable 'MQTT_HOST' is not set")
        if not mqtt_port:
            raise ValueError("Environment variable 'MQTT_PORT' is not set")
        if not mqtt_topic:
            raise ValueError("Environment variable 'MQTT_TOPIC' is not set")
        if not detection_areas:
            raise ValueError("Environment variable 'DETECTION_AREAS' is not set")

        return cls(
            rtsp_stream=rtsp_stream,
            mqtt_host=mqtt_host,
            mqtt_port=int(mqtt_port),
            mqtt_topic=mqtt_topic,
            tracking_areas=detection_areas_from_json(detection_areas),
        )


def detect_and_track(rtsp_stream: str) -> Iterator[Results]:
    """Detect and track objects in a video stream."""
    # TODO: make this more fault tolerant when the stream is not available. Don't crash
    model = YOLO("yolov8n-seg.pt")
    if DEBUG:
        return model.track(
            rtsp_stream, stream=True, verbose=False, classes=0, show=True
        )

    return model.track(rtsp_stream, stream=True, verbose=False, classes=0)


def filter_valid_objects_from_results(results: Results) -> Iterator[DetectedObject]:
    if results.boxes and results.masks:
        for box, mask in zip(results.boxes, results.masks):  # type: ignore fallback to '__get_item__'
            object = DetectedObject.from_box_mask_result(box, mask, results)
            if object.id is None:
                continue
            yield object


def publish_event_message(
    mqtt_client: MqttClient,
    root_topic: str,
    area: TrackingArea,
    object: DetectedObject,
    timestamp_sec: int,
    in_area: bool,
) -> None:
    message = {
        "object_id": object.id,
        "in_area": in_area,
        "timestamp_sec": timestamp_sec,
        "class_id": object.class_id,
        "class_name": object.class_name,
        "detection_confidence": object.detection_confidence,
    }
    mqtt_client.publish(
        f"{root_topic}/{area.tag}/events",
        json.dumps(message, indent=2),
    )


def publish_count_message(
    mqtt_client: MqttClient,
    root_topic: str,
    area: TrackingArea,
    object_count: int,
    timestamp_sec: int,
) -> None:
    message = {
        "object_count": object_count,
        "timestamp_sec": timestamp_sec,
    }
    mqtt_client.publish(
        f"{root_topic}/{area.tag}/count",
        json.dumps(message, indent=2),
    )


def analyze_results_and_publish(
    results_stream: Iterable[Results],
    tracking_areas: list[TrackingArea],
    mqtt_client: MqttClient,
    mqtt_root_topic: str,
) -> None:
    per_area_object_record: dict[str, set[DetectedObject]] = {
        k.tag: set() for k in tracking_areas
    }
    per_area_object_count: dict[str, int] = {k.tag: 0 for k in tracking_areas}

    for area, results in product(tracking_areas, results_stream):
        inference_timestamp_sec = int(time.time())
        previous_frame_object_record = per_area_object_record[area.tag]
        this_frame_object_record: set[DetectedObject] = set()

        for object in filter_valid_objects_from_results(results):
            if area_contains_object(area, object) is False:
                continue
            this_frame_object_record.add(object)

            if object not in previous_frame_object_record:
                per_area_object_record[area.tag].add(object)
                publish_event_message(
                    mqtt_client,
                    mqtt_root_topic,
                    area,
                    object,
                    inference_timestamp_sec,
                    in_area=True,
                )

        no_longer_in_area = previous_frame_object_record.difference(
            this_frame_object_record
        )
        previous_frame_object_record.intersection_update(this_frame_object_record)
        for object in no_longer_in_area:
            publish_event_message(
                mqtt_client,
                mqtt_root_topic,
                area,
                object,
                inference_timestamp_sec,
                in_area=False,
            )

        this_frame_object_count = len(this_frame_object_record)
        if this_frame_object_count != per_area_object_count[area.tag]:
            per_area_object_count[area.tag] = this_frame_object_count
            publish_count_message(
                mqtt_client,
                mqtt_root_topic,
                area,
                this_frame_object_count,
                inference_timestamp_sec,
            )


if __name__ == "__main__":
    if DEBUG:
        app_config = AppConfig(
            rtsp_stream=DEBUG_RTSP_STREAM,
            mqtt_host=DEBUG_MQTT_HOST,
            mqtt_port=DEBUG_MQTT_PORT,
            mqtt_topic=DEBUG_MQTT_TOPIC,
            tracking_areas=[
                TrackingArea(DEBUG_DETECTION_AREA_TAG, DEBUG_DETECTION_AREA_POLYGON)
            ],
        )
    else:
        app_config = AppConfig.from_env()

    if DEBUG:
        mqtt_client = DummyMqttClient()
    else:
        mqtt_client = PahoMqttClient()
        mqtt_client.connect(app_config.mqtt_host, app_config.mqtt_port)

    results = detect_and_track(app_config.rtsp_stream)

    analyze_results_and_publish(
        results, app_config.tracking_areas, mqtt_client, app_config.mqtt_topic
    )
