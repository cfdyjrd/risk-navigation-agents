"""Pure offline corridor estimator tests; ROS and hardware are not imported."""

import json
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest

from g1_corridor_perception import estimate_corridor, pointcloud_xyz, transform_points
from robot_interface import RobotInterfaceError, RobotObservation
from s1_experiment import require_corridor_geometry


ROOT = Path(__file__).resolve().parent


def config():
    data = json.loads((ROOT / "configs" / "s1.example.json").read_text(encoding="utf-8"))
    data["geometry"].update(
        robot_effective_width_m=.60,
        robot_left_extent_m=.30,
        robot_right_extent_m=.30,
    )
    data["layouts"]["A"]["measured_width_m"] = 1.40
    data["layouts"]["B"]["measured_width_m"] = .90
    for name, offset in (("A", .70), ("B", .45)):
        data["layouts"][name].update(
            left_inner_offset_m=offset,
            right_inner_offset_m=offset,
            measured_box_length_m=1.20,
            measured_box_height_m=1.70,
            measurement_note=f"offline fixture {name}",
            box_shape_and_fixing_note="offline rectangular fixed fixture",
            foot_envelope_relation_note="offline envelope relation",
        )
    data["perception"]["onsite_validation"].update(
        status="onsite_verified", operator="test", verified_at="2026-09-10T00:00:00Z",
        tf_method="fixture", robot_envelope_method="fixture", box_geometry_method="fixture",
    )
    data["_config_path"] = str(ROOT / "configs" / "s1.test.json")
    return data


def walls(include_obstacle=False):
    points = []
    for index in range(8):
        x = .30 + index * .12
        for point_index in range(5):
            jitter = (point_index - 2) * .001
            points.append((x, .45 + jitter, .40 + point_index * .03))
            points.append((x, -.45 + jitter, .40 + point_index * .03))
    if include_obstacle:
        points.extend((.55 + index * .002, 0.0, .5) for index in range(4))
    return points


class CorridorPerceptionTests(unittest.TestCase):
    def test_estimates_asymmetric_envelope_clearance_conservatively(self):
        result = estimate_corridor(walls(), config(), "B")
        self.assertTrue(result["validated"])
        self.assertTrue(result["corridor_geometry_valid"])
        self.assertAlmostEqual(result["corridor_width_m"], .90, places=2)
        self.assertAlmostEqual(result["left_envelope_clearance_m"], .15, places=2)
        self.assertAlmostEqual(result["right_envelope_clearance_m"], .15, places=2)
        self.assertLess(result["minimum_envelope_clearance_m"], .15)
        self.assertAlmostEqual(result["corridor_wall_start_m"], .35, places=2)
        self.assertGreaterEqual(
            result["corridor_wall_end_m"] - result["corridor_wall_start_m"], .7
        )
        self.assertFalse(result["obstacle_detected"])

    def test_detects_cluster_inside_swept_envelope(self):
        result = estimate_corridor(walls(include_obstacle=True), config(), "B")
        self.assertTrue(result["obstacle_detected"])
        self.assertAlmostEqual(result["obstacle_distance_m"], .55, places=2)

    def test_one_local_wall_pinch_is_not_hidden_by_the_median(self):
        points = walls()
        points = [
            (x, .38 + (y - .45), z) if .65 <= x <= .75 and y > 0 else (x, y, z)
            for x, y, z in points
        ]
        result = estimate_corridor(points, config(), "B")
        self.assertTrue(result["corridor_geometry_valid"])
        self.assertAlmostEqual(result["left_envelope_clearance_m"], .08, places=2)
        self.assertLess(result["minimum_envelope_clearance_m"], .10)
        self.assertGreaterEqual(result["critical_wall_bin_x_m"], .65)
        self.assertLessEqual(result["critical_wall_bin_x_m"], .75)

    def test_missing_paired_wall_support_never_invents_geometry(self):
        one_wall = [point for point in walls() if point[1] > 0]
        result = estimate_corridor(one_wall, config(), "B")
        self.assertFalse(result["corridor_geometry_valid"])
        self.assertNotIn("minimum_envelope_clearance_m", result)

    def test_rigid_transform_is_applied(self):
        points = list(transform_points([(1., 0., 0.)], (1., 2., 3.), (0., 0., 0., 1.)))
        self.assertEqual(points, [(2., 2., 3.)])

    def test_pointcloud_parser_respects_endianness_and_row_padding(self):
        message = SimpleNamespace(
            fields=[
                SimpleNamespace(name=name, offset=index * 4, datatype=7, count=1)
                for index, name in enumerate(("x", "y", "z"))
            ],
            is_bigendian=True,
            point_step=12,
            row_step=16,
            width=1,
            height=2,
            data=(
                struct.pack(">fff", 1., 2., 3.) + b"pad!"
                + struct.pack(">fff", 4., 5., 6.) + b"pad!"
            ),
        )
        self.assertEqual(
            list(pointcloud_xyz(message, max_points=10)),
            [(1., 2., 3.), (4., 5., 6.)],
        )

    def test_preflight_checks_registered_corridor_entrance(self):
        settings = config()
        environment = estimate_corridor(walls(), settings, "B")
        observation = RobotObservation(
            robot={"type": "g1", "width_m": .6, "battery_percent": 80},
            environment=environment,
            timestamp="2026-09-10T00:00:00+08:00",
        )
        require_corridor_geometry(
            observation, settings, "B", expected_wall_start_m=.30
        )
        with self.assertRaises(RobotInterfaceError):
            require_corridor_geometry(
                observation, settings, "B", expected_wall_start_m=1.50
            )


if __name__ == "__main__":
    unittest.main()
