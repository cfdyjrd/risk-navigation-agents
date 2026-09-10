import unittest
from datetime import datetime, timezone
from unittest.mock import patch, Mock

from execution_bridge import SafeExecutionBridge
from robot_interface import RobotObservation, ExecutionReceipt
from robot_task_loop import RobotTaskLoop, make_memory_decider
from test_execution_bridge import ACTIONS, optimizer_decision


def now():
    return datetime.now(timezone.utc).isoformat()


class Robot:
    def __init__(self):
        self.actions = []
        self.stops = []
        self.stale = None
        self.stationary = True
        self.distance = 1.0

    def observe(self):
        return RobotObservation(
            {'type': 'go2', 'width_m': .5, 'battery_percent': 80},
            {'corridor_width_m': 1.3, 'obstacle_detected': True,
             'obstacle_distance_m': self.distance, 'observation_confidence': .9},
            self.stale or now(), metadata={'stationary': self.stationary})

    def execute(self, action):
        name = action['name']
        self.actions.append(name)
        return ExecutionReceipt('aborted' if name == 'observe_again' else 'success',
                                'submitted', now(), now(), {'requested_action': name})

    def emergency_stop(self, reason):
        self.stops.append(reason)
        return ExecutionReceipt('aborted', reason, now(), now())


class LoopTests(unittest.TestCase):
    def loop(self, robot, decide, **kwargs):
        return RobotTaskLoop(SafeExecutionBridge(robot, available_actions=ACTIONS), decide,
                             is_stationary=lambda o: o.metadata.get('stationary') is True,
                             goal_reached=kwargs.pop('goal_reached', lambda o, t: False),
                             poll_interval_s=.001, observation_timeout_s=.02, **kwargs)

    def test_reobserve_gets_new_frames_and_replans(self):
        robot = Robot()
        frames = []
        def decide(s):
            frames.append(s['observation']['timestamp'])
            return optimizer_decision('observe_again' if len(frames) == 1 else 'slow_down')
        result = self.loop(robot, decide, goal_reached=lambda o,t: 'slow_down' in robot.actions).run({'goal':'G'})
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(robot.actions, ['observe_again', 'slow_down'])
        self.assertGreater(frames[1], frames[0])
        self.assertFalse(robot.stops)

    def test_stale_frames_after_stop_never_reach_planner(self):
        robot = Robot()
        calls = []
        def decide(s):
            calls.append(s)
            robot.stale = s['observation']['timestamp']
            return optimizer_decision('observe_again')
        result = self.loop(robot, decide).run({'goal':'G'})
        self.assertEqual(result['status'], 'fail_closed')
        self.assertEqual(len(calls), 1)
        self.assertEqual(robot.actions, ['observe_again'])
        self.assertTrue(robot.stops)

    def test_unknown_or_moving_state_cannot_plan(self):
        robot = Robot(); robot.stationary = False
        decide = Mock()
        result = self.loop(robot, decide).run({'goal':'G'})
        self.assertEqual(result['status'], 'fail_closed')
        decide.assert_not_called()

    def test_retry_limit(self):
        robot = Robot()
        result = self.loop(robot, lambda s: optimizer_decision('observe_again'), max_reobservations=2).run({'goal':'G'})
        self.assertEqual(robot.actions, ['observe_again', 'observe_again'])
        self.assertIn('limit', result['error'])

    def test_new_hazard_still_guarded(self):
        robot = Robot()
        def decide(s):
            robot.distance = .2
            return optimizer_decision('move_forward')
        result = self.loop(robot, decide).run({'goal':'G'})
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(robot.actions, ['safe_stop'])

    def test_human_request_does_not_resume(self):
        robot = Robot()
        result = self.loop(robot, lambda s: optimizer_decision('ask_human')).run({'goal':'G'})
        self.assertEqual(result['status'], 'needs_human')
        self.assertEqual(robot.actions, ['ask_human'])

    def test_planner_failure_stops(self):
        robot = Robot()
        result = self.loop(robot, Mock(side_effect=RuntimeError('LLM unavailable'))).run({'goal':'G'})
        self.assertEqual(result['status'], 'fail_closed')
        self.assertFalse(robot.actions)
        self.assertTrue(robot.stops)

    def test_step_limit_does_not_claim_arrival(self):
        robot = Robot()
        result = self.loop(robot, lambda s: optimizer_decision('slow_down'), max_steps=1).run({'goal':'G'})
        self.assertEqual(result['status'], 'fail_closed')
        self.assertTrue(robot.stops)

    def test_memory_callback_retrieves_each_time(self):
        store = Mock(); store.retrieve_memory_cards.return_value = {'items': []}
        rules = Mock(); rules.retrieve_matching.return_value = []
        audit = Mock()
        scenario = {'available_actions': ACTIONS}
        with patch('run_three_agents.run_advocate', return_value=({}, {})), \
             patch('run_three_agents.run_critic', return_value=({}, {})), \
             patch('run_three_agents.decide', return_value=({}, {})), \
             patch('run_three_agents.run_post_deliberation_optimizer', return_value=({}, optimizer_decision('safe_stop'))):
            callback = make_memory_decider(store, rules, Mock(), audit_sink=audit)
            callback(scenario); callback(scenario)
        self.assertEqual(store.retrieve_memory_cards.call_count, 6)
        self.assertEqual(rules.retrieve_matching.call_count, 2)
        self.assertEqual(audit.call_count, 2)


if __name__ == '__main__':
    unittest.main()
