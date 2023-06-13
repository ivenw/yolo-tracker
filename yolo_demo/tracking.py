from dataclasses import dataclass
from typing import Optional, Union, cast

import numpy as np
from shapely.geometry import Point, Polygon
from torch import Tensor
from typing_extensions import Self
from ultralytics.yolo.engine.results import Boxes, Masks, Results


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
    bounding_box_normalized: Union[Tensor, np.ndarray]
    segment_normalized: np.ndarray

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
            bounding_box_normalized=box.xyxyn[0],  # type: ignore
            segment_normalized=mask.xyn[0],
        )

    @property
    def max_segment_y_point(self) -> Point:
        raise DeprecationWarning("Will be removed in the future.")
        point_coords_idx = np.where(
            self.segment_normalized == np.max(self.segment_normalized[:, 1])
        )[0]
        point_coords = self.segment_normalized[point_coords_idx]
        return Point(point_coords[0][0], point_coords[0][1])

    @property
    def feet_segment(self) -> Union[Polygon, None]:
        percent_to_keep = 0.1
        min_y = np.min(self.segment_normalized[:, 1])
        max_y = np.max(self.segment_normalized[:, 1])
        relative_height = max_y - min_y
        min_y_cutoff = max_y - relative_height * percent_to_keep
        points_to_keep = np.where(self.segment_normalized[:, 1] > min_y_cutoff)
        feet_box_points = self.segment_normalized[points_to_keep]

        if len(feet_box_points) < 3:
            return None

        return Polygon(feet_box_points)


def object_intersects_area(
    object: DetectedObject, detection_area: TrackingArea
) -> bool:
    """Check if a detected object intersects a detection area."""
    if object.feet_segment is None:
        return False

    return object.feet_segment.intersects(detection_area.polygon)


def area_contains_object(
    detection_area: TrackingArea, detected_object: DetectedObject
) -> bool:
    """Check if a detection area contains a detected object.

    Assumes that detection area is in plane of an even floor and that the object is
    standing on the floor.
    """
    raise DeprecationWarning("Will be removed in the future.")
    return detection_area.polygon.contains(detected_object.max_segment_y_point)
