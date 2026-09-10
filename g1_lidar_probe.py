"""Read-only PointCloud2 metadata probe; raw range is NOT obstacle clearance."""
import argparse
import json
import math
from pathlib import Path
import struct
import sys
import time

from g1_dds_diagnostic import _run_supervised, _progress


def summarize_cloud(msg):
    fields = {field.name: field for field in msg.fields}
    xyz = [fields[key] for key in ("x", "y", "z")]
    endian = ">" if msg.is_bigendian else "<"
    readers = []
    for field in xyz:
        if field.count != 1 or field.datatype not in (7, 8):
            raise ValueError("x/y/z must be scalar FLOAT32 or FLOAT64")
        reader = struct.Struct(endian + ("f" if field.datatype == 7 else "d"))
        if field.offset < 0 or field.offset + reader.size > msg.point_step:
            raise ValueError("invalid PointCloud2 field offset")
        readers.append((reader, field.offset))
    if msg.row_step < msg.width * msg.point_step or len(msg.data) < msg.height * msg.row_step:
        raise ValueError("invalid PointCloud2 data layout")
    finite, nearest = 0, None
    for row in range(msg.height):
        for col in range(msg.width):
            offset = row * msg.row_step + col * msg.point_step
            values = [reader.unpack_from(msg.data, offset + extra)[0] for reader, extra in readers]
            if not all(math.isfinite(value) for value in values):
                continue
            finite += 1
            distance = math.sqrt(sum(value * value for value in values))
            if distance > 0:
                nearest = distance if nearest is None else min(nearest, distance)
    return {"frame_id": msg.header.frame_id,
            "source_stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
            "points": msg.width * msg.height, "finite_points": finite,
            "nearest_raw_return_m": nearest,
            "validated_obstacle_perception": False,
            "note": "Raw sensor-frame returns include floor/body; no robot-frame transform or clearance verification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/utlidar/cloud_livox_mid360")
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.duration <= 30:
        parser.error("duration must be within 1–30 seconds")
    if not args.worker:
        return _run_supervised([sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                               "--topic", args.topic, "--duration", str(args.duration)], args.duration + 10)
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    rclpy.init()
    node = rclpy.create_node("risk_navigation_g1_lidar_probe")
    latest, counts = {}, {"frames": 0}

    def receive(msg):
        counts["frames"] += 1
        # Summarize only the first frame and then at most once per second.
        if time.monotonic() - counts.get("last_summary", 0) < 1:
            return
        counts["last_summary"] = time.monotonic()
        try:
            latest.clear()
            latest.update(summarize_cloud(msg))
        except Exception as exc:
            latest["error"] = str(exc)

    subscription = node.create_subscription(PointCloud2, args.topic, receive, qos_profile_sensor_data)
    _progress("只读订阅点云：" + args.topic)
    end = time.monotonic() + args.duration
    try:
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.1)
        print(json.dumps({"topic": args.topic, "frames": counts["frames"], "latest": latest,
                          "motion_commands_sent": 0, "motion_ready": False}, indent=2, allow_nan=False), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0 if counts["frames"] and latest and "error" not in latest else 1


if __name__ == "__main__":
    raise SystemExit(main())
