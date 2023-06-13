from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from typing_extensions import Self
from ultralytics.yolo.engine.results import Results

from yolo_demo.tracking import DetectedObject, TrackingArea


@dataclass
class FrameAnnotator:
    result: Results

    def __post_init__(self) -> None:
        self._frame = self.result.plot()
        self._xy_dimensions = self._frame.shape[:2][::-1]  # type: ignore

    def annotate_tracking_areas(self, tracking_areas: Iterable[TrackingArea]) -> Self:
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        line_thickness = int(0.003 * self._xy_dimensions[0])

        for area, color in zip(tracking_areas, colors):
            polygon = np.array(area.polygon.boundary.coords, np.float32)
            polygon = np.multiply(polygon, self._xy_dimensions).astype(np.int32)
            cv2.polylines(
                self._frame,
                pts=[polygon],
                isClosed=True,
                color=color,
                thickness=line_thickness,
            )
            text = f"tag: {area.tag}"
            # TODO: make this scale with image size
            font_scale = 3
            font_thickness = 5
            text_width, text_height = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )[0]
            # TODO: Place text at left bottom of polygon
            p1 = polygon[0]
            p2 = (polygon[0][0] + text_width, polygon[0][1] - text_height)
            cv2.rectangle(self._frame, p1, p2, color, -1)
            cv2.putText(
                self._frame,
                text=text,
                org=polygon[0],
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3,
                color=(255, 255, 255),
                thickness=5,
                lineType=cv2.LINE_AA,
            )
        return self

    def annotate_object_tracking_position(
        self, objects: Iterable[DetectedObject]
    ) -> Self:
        point_size = int(0.02 * self._xy_dimensions[0])
        line_thickness = int(0.003 * self._xy_dimensions[0])

        for object in objects:
            polygon = np.array(object.feet_segment.boundary.coords, np.float32)
            polygon = np.multiply(polygon, self._xy_dimensions).astype(np.int32)
            cv2.polylines(
                self._frame,
                pts=[polygon],
                isClosed=True,
                color=(0, 255, 0),
                thickness=line_thickness,
            )

        return self

    def annotate_object_tracking_position_legacy(
        self, objects: Iterable[DetectedObject]
    ) -> Self:
        raise DeprecationWarning("Use annotate_object_tracking_position instead")
        point_size = int(0.02 * self._xy_dimensions[0])

        for object in objects:
            point_coords = (
                np.array(object.max_segment_y_point.coords, np.float32)
                * self._xy_dimensions
            ).astype(np.int32)[0]
            cv2.circle(
                self._frame,
                center=point_coords,
                radius=point_size,
                color=(255, 0, 0),
                thickness=-1,
            )
        return self

    def to_ndarrary(self) -> np.ndarray:
        return self._frame

    def show(self) -> None:
        cv2.imshow("annotated", self._frame)
        if cv2.waitKey(1) == 27:  # ESC key
            exit()
