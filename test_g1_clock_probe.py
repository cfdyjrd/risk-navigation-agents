import unittest
from g1_clock_probe import offset_interval


class ClockTests(unittest.TestCase):
    def test_known_remote_delay_is_bounded(self):
        result = offset_interval(30_000_000_000, 30_020_000_000, 5_010_000_000)
        self.assertAlmostEqual(result["round_trip_s"], .02)
        self.assertAlmostEqual(result["local_minus_remote_min_s"], 24.99)
        self.assertAlmostEqual(result["local_minus_remote_max_s"], 25.01)

    def test_reversed_or_invalid_clock_rejected(self):
        for values in ((10, 9, 5), (0, 10, 5), (True, 10, 5)):
            with self.assertRaises(ValueError):
                offset_interval(*values)


if __name__ == "__main__":
    unittest.main()
