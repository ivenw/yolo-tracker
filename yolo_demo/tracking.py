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
        # TODO: instead of a point, this should be a line between left and right most max y points
        point_coords_idx = np.where(
            self.segment_normalized == np.max(self.segment_normalized[:, 1])
        )[0]
        point_coords = self.segment_normalized[point_coords_idx]
        return Point(point_coords[0][0], point_coords[0][1])
