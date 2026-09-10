"""Read-only calibrated PointCloud2 corridor perception for G1 S1 trials.

The node never imports a locomotion client.  It transforms cloud points through
the live ROS TF tree into ``base_link``, estimates both inner wall surfaces,
subtracts the measured asymmetric robot envelope and estimator uncertainty,
and atomically writes a short-lived JSON cache.  Missing TF, weak wall support,
or an incomplete onsite attestation produces an invalid cache, never invented
clearance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import sys
import time
from typing import Iterable, Iterator

from g1_dds_diagnostic import _progress
from g1_state_collector import StreamLiveness, atomic_json
from s1_config import LAYOUTS, load_s1_config, resolve_config_path, trial_geometry


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("quantile of empty sequence")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _median(values: list[float]) -> float:
    return _quantile(values, 0.5)


def _mad(values: list[float], centre: float) -> float:
    return _median([abs(value - centre) for value in values])


def _theil_sen_slope(x_values: list[float], y_values: list[float]) -> float:
    slopes = [
        (y_values[right] - y_values[left]) / (x_values[right] - x_values[left])
        for left in range(len(x_values))
        for right in range(left + 1, len(x_values))
        if x_values[right] > x_values[left]
    ]
    if not slopes:
        raise ValueError("at least two distinct wall bins are required")
    return _median(slopes)


def quaternion_rotation(quaternion: Iterable[float]) -> list[list[float]]:
    values = list(quaternion)
    if len(values) != 4 or any(not _finite(value) for value in values):
        raise ValueError("TF quaternion must contain four finite values")
    x, y, z, w = (float(value) for value in values)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9:
        raise ValueError("TF quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def transform_points(
    points: Iterable[tuple[float, float, float]],
    translation: Iterable[float],
    quaternion: Iterable[float],
) -> Iterator[tuple[float, float, float]]:
    shift = list(translation)
    if len(shift) != 3 or any(not _finite(value) for value in shift):
        raise ValueError("TF translation must contain three finite values")
    rotation = quaternion_rotation(quaternion)
    for point in points:
        if len(point) != 3 or any(not _finite(value) for value in point):
            continue
        x, y, z = point
        yield (
            rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z + shift[0],
            rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z + shift[1],
            rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z + shift[2],
        )


def pointcloud_xyz(message, *, max_points: int) -> Iterator[tuple[float, float, float]]:
    """Yield a deterministic bounded sample of PointCloud2 xyz coordinates."""
    fields = {field.name: field for field in message.fields}
    try:
        xyz_fields = [fields[name] for name in ("x", "y", "z")]
    except KeyError as exc:
        raise ValueError("PointCloud2 lacks x/y/z fields") from exc
    endian = ">" if message.is_bigendian else "<"
    readers = []
    for field in xyz_fields:
        if field.count != 1 or field.datatype not in (7, 8):
            raise ValueError("x/y/z must be scalar FLOAT32 or FLOAT64")
        reader = struct.Struct(endian + ("f" if field.datatype == 7 else "d"))
        if field.offset < 0 or field.offset + reader.size > message.point_step:
            raise ValueError("invalid PointCloud2 field offset")
        readers.append((reader, field.offset))
    total = int(message.width) * int(message.height)
    if total < 1 or message.point_step <= 0:
        raise ValueError("empty PointCloud2")
    if message.row_step < message.width * message.point_step:
        raise ValueError("invalid PointCloud2 row_step")
    if len(message.data) < message.height * message.row_step:
        raise ValueError("truncated PointCloud2 data")
    stride = max(1, math.ceil(total / max_points))
    flat_index = 0
    for row in range(message.height):
        for column in range(message.width):
            if flat_index % stride == 0:
                offset = row * message.row_step + column * message.point_step
                values = tuple(
                    reader.unpack_from(message.data, offset + extra)[0]
                    for reader, extra in readers
                )
                if all(math.isfinite(value) for value in values):
                    yield values
            flat_index += 1


def estimate_corridor(
    points_base_link: Iterable[tuple[float, float, float]],
    config: dict,
    layout: str,
) -> dict:
    """Estimate conservative side clearances from already transformed points."""
    if layout not in LAYOUTS:
        raise ValueError("layout must be A or B")
    roi = config["perception"]["roi"]
    settings = config["perception"]["estimator"]
    geometry = config["geometry"]
    trial = trial_geometry(config, layout)
    x_min, x_max = float(roi["forward_min_m"]), float(roi["forward_max_m"])
    z_min, z_max = float(roi["height_min_m"]), float(roi["height_max_m"])
    roi_points = [
        (float(x), float(y), float(z))
        for x, y, z in points_base_link
        if all(_finite(value) for value in (x, y, z))
        and x_min <= x <= x_max and z_min <= z <= z_max
    ]
    minimum_roi = int(settings["minimum_total_roi_points"])
    observation_confidence = min(1.0, len(roi_points) / minimum_roi) if minimum_roi else 0.0
    result = {
        "validated": len(roi_points) >= minimum_roi,
        "observation_confidence": round(observation_confidence, 4),
        "roi_point_count": len(roi_points),
        "corridor_geometry_valid": False,
        "corridor_geometry_confidence": 0.0,
        "obstacle_detected": False,
        "obstacle_distance_m": None,
    }
    if not result["validated"]:
        result["validation_error"] = "insufficient points in calibrated ROI"
        return result

    left_extent = float(geometry["robot_left_extent_m"])
    right_extent = float(geometry["robot_right_extent_m"])
    lateral_margin = float(settings["obstacle_lateral_margin_m"])
    obstruction = [
        point for point in roi_points
        if point[0] >= float(settings["obstacle_forward_min_m"])
        and -right_extent - lateral_margin <= point[1] <= left_extent + lateral_margin
    ]
    minimum_obstacle = int(settings["minimum_obstacle_points"])
    if len(obstruction) >= minimum_obstacle:
        result["obstacle_detected"] = True
        result["obstacle_distance_m"] = round(
            max(0.0, _quantile([point[0] for point in obstruction], 0.05)), 4
        )

    expected_width = float(trial["measured_corridor_width_m"])
    search_margin = float(settings["wall_search_margin_m"])
    inner_floor = max(0.05, min(left_extent, right_extent) * 0.5)
    outer_limit = expected_width / 2 + search_margin
    bin_size = float(settings["wall_bin_size_m"])
    left_bins: dict[int, list[float]] = {}
    right_bins: dict[int, list[float]] = {}
    for x, y, _z in roi_points:
        index = int(math.floor((x - x_min) / bin_size))
        if inner_floor <= y <= outer_limit:
            left_bins.setdefault(index, []).append(y)
        elif -outer_limit <= y <= -inner_floor:
            right_bins.setdefault(index, []).append(y)
    minimum_per_bin = int(settings["minimum_points_per_wall_bin"])
    common = sorted(set(left_bins).intersection(right_bins))
    qualified_bins = [
        index for index in common
        if len(left_bins[index]) >= minimum_per_bin
        and len(right_bins[index]) >= minimum_per_bin
    ]
    left_boundaries = [
        _quantile(left_bins[index], 0.10)
        for index in qualified_bins
    ]
    right_boundaries = [
        _quantile(right_bins[index], 0.90)
        for index in qualified_bins
    ]
    minimum_bins = int(settings["minimum_wall_bins"])
    result["wall_support"] = {
        "common_bins": len(left_boundaries),
        "left_candidate_points": sum(len(values) for values in left_bins.values()),
        "right_candidate_points": sum(len(values) for values in right_bins.values()),
    }
    if len(left_boundaries) < minimum_bins:
        result["geometry_error"] = "insufficient paired wall bins"
        return result

    result["corridor_wall_start_m"] = round(
        x_min + (min(qualified_bins) + 0.5) * bin_size, 4
    )
    result["corridor_wall_end_m"] = round(
        x_min + (max(qualified_bins) + 0.5) * bin_size, 4
    )

    x_centres = [x_min + (index + 0.5) * bin_size for index in qualified_bins]
    corridor_centres = [
        (left_boundary + right_boundary) / 2
        for left_boundary, right_boundary in zip(left_boundaries, right_boundaries)
    ]
    result["corridor_heading_error_rad"] = round(
        math.atan(_theil_sen_slope(x_centres, corridor_centres)), 6
    )

    left_wall = _median(left_boundaries)
    right_wall = _median(right_boundaries)
    left_mad = _mad(left_boundaries, left_wall)
    right_mad = _mad(right_boundaries, right_wall)
    median_width = left_wall - right_wall
    width_error = abs(median_width - expected_width)
    max_mad = float(settings["maximum_wall_mad_m"])
    width_tolerance = float(settings["wall_width_tolerance_m"])
    bin_score = min(1.0, len(left_boundaries) / max(minimum_bins * 2, 1))
    width_score = max(0.0, 1.0 - width_error / width_tolerance)
    dispersion_score = max(0.0, 1.0 - max(left_mad, right_mad) / max_mad)
    geometry_confidence = min(observation_confidence, bin_score, width_score, dispersion_score)
    result["corridor_geometry_confidence"] = round(geometry_confidence, 4)
    result["wall_estimate"] = {
        "basis": "median_paired_wall_bins_for_quality_gate",
        "left_inner_y_m": round(left_wall, 4),
        "right_inner_y_m": round(right_wall, 4),
        "median_width_m": round(median_width, 4),
        "left_mad_m": round(left_mad, 4),
        "right_mad_m": round(right_mad, 4),
        "expected_width_m": round(expected_width, 4),
        "width_error_m": round(width_error, 4),
    }
    if (
        width_error > width_tolerance
        or max(left_mad, right_mad) > max_mad
        or geometry_confidence < float(settings["minimum_geometry_confidence"])
    ):
        result["geometry_error"] = "wall estimate failed width, dispersion, or confidence gate"
        return result

    uncertainty = float(settings["measurement_uncertainty_floor_m"]) + max(
        left_mad, right_mad
    )
    paired_clearances = [
        (left_boundary - left_extent, -right_boundary - right_extent)
        for left_boundary, right_boundary in zip(left_boundaries, right_boundaries)
    ]
    critical_index = min(
        range(len(paired_clearances)),
        key=lambda index: min(paired_clearances[index]),
    )
    left_critical_wall = left_boundaries[critical_index]
    right_critical_wall = right_boundaries[critical_index]
    left_clearance, right_clearance = paired_clearances[critical_index]
    measured_width = left_critical_wall - right_critical_wall
    conservative = min(left_clearance, right_clearance) - uncertainty
    result.update({
        "corridor_geometry_valid": True,
        "corridor_width_m": round(measured_width, 4),
        "left_envelope_clearance_m": round(left_clearance, 4),
        "right_envelope_clearance_m": round(right_clearance, 4),
        "clearance_uncertainty_m": round(uncertainty, 4),
        "minimum_envelope_clearance_m": round(conservative, 4),
        "corridor_center_offset_m": round(
            (left_critical_wall + right_critical_wall) / 2, 4
        ),
        "critical_wall_bin_x_m": round(x_centres[critical_index], 4),
        "clearance_basis": "worst_side_at_most_constrained_paired_wall_bin",
    })
    return result


def _invalid_record(config: dict, layout: str, error: Exception | str) -> dict:
    return {
        "producer": "g1_corridor_perception",
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_basis": "local_receipt_after_sensor_message",
        "frame_id": "base_link",
        "source_frame": config["perception"]["sensor_frame"],
        "calibration_id": config["perception"]["calibration_id"],
        "layout": layout,
        "validated": False,
        "corridor_geometry_valid": False,
        "obstacle_detected": True,
        "obstacle_distance_m": 0.0,
        "observation_confidence": 0.0,
        "error": str(error),
        "motion_commands_sent": 0,
    }


def run_ros(config: dict, layout: str, output: Path, *, calibration_check: bool = False) -> int:
    import rclpy
    from rclpy.duration import Duration
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import PointCloud2
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = rclpy.create_node("risk_navigation_s1_corridor_perception")
    tf_buffer = Buffer()
    listener = TransformListener(tf_buffer, node)
    counts = {"received": 0, "valid": 0, "invalid": 0}
    # One persistent tracker per subscription is required to prove that source
    # timestamps advance across messages.  Recreating it per callback would
    # make every frame permanently unverified.
    liveness_tracker = StreamLiveness()
    expected_source = config["perception"]["sensor_frame"].lstrip("/")
    base_frame = config["perception"]["base_frame"]
    settings = config["perception"]["estimator"]

    def receive(message):
        counts["received"] += 1
        try:
            received_wall_ns = time.time_ns()
            received_monotonic_ns = time.monotonic_ns()
            source_ns = (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            )
            liveness = liveness_tracker.observe(
                source_ns, received_wall_ns, received_monotonic_ns
            )
            actual_source = message.header.frame_id.lstrip("/")
            if actual_source != expected_source:
                raise ValueError(
                    f"unexpected cloud frame {message.header.frame_id!r}; expected {expected_source!r}"
                )
            transform = tf_buffer.lookup_transform(
                base_frame,
                message.header.frame_id,
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=0.2),
            )
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            sensor_points = pointcloud_xyz(message, max_points=int(settings["max_points"]))
            points = transform_points(
                sensor_points,
                (translation.x, translation.y, translation.z),
                (rotation.x, rotation.y, rotation.z, rotation.w),
            )
            estimate = estimate_corridor(points, config, layout)
            estimator_valid = estimate["validated"]
            estimate["source_liveness"] = liveness
            estimate["validated"] = estimator_valid and liveness["verified"]
            if estimator_valid and not liveness["verified"]:
                estimate["validation_error"] = "point-cloud source liveness is not yet verified"
            if calibration_check:
                estimate["estimator_candidate_valid"] = estimate["validated"]
                estimate["validated"] = False
                estimate["calibration_check_only"] = True
                estimate["validation_error"] = (
                    "calibration-check output is never authorized for motion"
                )
            received_at = datetime.fromtimestamp(
                received_wall_ns / 1e9, timezone.utc
            ).isoformat()
            record = {
                "producer": "g1_corridor_perception",
                "schema_version": 1,
                "timestamp": received_at,
                "timestamp_basis": "local_receipt_after_sensor_message",
                "received_at": received_at,
                "frame_id": base_frame,
                "source_frame": message.header.frame_id,
                "source_stamp": {
                    "sec": int(message.header.stamp.sec),
                    "nanosec": int(message.header.stamp.nanosec),
                },
                "calibration_id": config["perception"]["calibration_id"],
                "layout": layout,
                "robot_envelope": {
                    "width_m": config["geometry"]["robot_effective_width_m"],
                    "left_extent_m": config["geometry"]["robot_left_extent_m"],
                    "right_extent_m": config["geometry"]["robot_right_extent_m"],
                },
                "onsite_validation": config["perception"]["onsite_validation"],
                "motion_commands_sent": 0,
                **estimate,
            }
            counts["valid" if record["validated"] else "invalid"] += 1
            atomic_json(output, record)
        except Exception as exc:
            counts["invalid"] += 1
            atomic_json(output, _invalid_record(config, layout, exc))

    subscription = node.create_subscription(
        PointCloud2, config["perception"]["topic"], receive, qos_profile_sensor_data
    )
    _progress(
        "S1只读通道感知已启动：%s -> %s；模式=%s；不会发送运动指令。"
        % (config["perception"]["topic"], output,
           "calibration_check" if calibration_check else "validated_runtime")
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        _progress("S1只读通道感知已停止。")
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps({**counts, "motion_commands_sent": 0}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--calibration-check", action="store_true",
        help="Estimate stationary geometry but force validated=false; never usable for motion",
    )
    args = parser.parse_args()
    try:
        config = load_s1_config(
            args.config, require_onsite=not args.calibration_check
        )
        output = (
            args.output.expanduser().resolve()
            if args.output
            else resolve_config_path(config, config["paths"]["perception_file"])
        )
        if not output.parent.is_dir() or output.suffix != ".json":
            raise ValueError("output must be a .json file in an existing directory")
        if output.exists():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
                if existing.get("producer") != "g1_corridor_perception":
                    raise ValueError("refusing to overwrite a cache owned by another producer")
            except (OSError, ValueError, AttributeError) as exc:
                raise ValueError(f"cannot reuse perception output: {exc}") from exc
        return run_ros(
            config, args.layout, output,
            calibration_check=args.calibration_check,
        )
    except Exception as exc:
        print(json.dumps({
            "status": "failed", "error": str(exc), "motion_commands_sent": 0
        }, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
