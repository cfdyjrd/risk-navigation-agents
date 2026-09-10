"""Bounded observe/deliberate/execute loop; hardware callbacks must be bounded."""
from copy import deepcopy
from datetime import datetime, timezone
import math
import time

from robot_interface import RobotInterfaceError, observation_to_scenario


def _timestamp(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise RobotInterfaceError('timestamps must include a timezone')
    return stamp.timestamp()


class RobotTaskLoop:
    """Replan from new, stationary observations, never resume an old action.

    is_stationary must use measured feedback, not command submission. The
    arrival callback must independently verify the goal. No probing movement
    or human answer is fabricated here. Callbacks must enforce their own I/O
    deadlines; synchronous Python cannot interrupt a blocked vendor SDK.
    """
    def __init__(self, bridge, decide, *, is_stationary, goal_reached,
                 max_steps=30, max_reobservations=3, observation_timeout_s=3.0,
                 max_observation_age_s=1.0, task_timeout_s=120.0,
                 poll_interval_s=0.05):
        for name, value in [('max_steps', max_steps),
                            ('max_reobservations', max_reobservations)]:
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for value in (observation_timeout_s, max_observation_age_s,
                      task_timeout_s, poll_interval_s):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('timeouts and poll interval must be positive and finite')
        self.bridge, self.decide = bridge, decide
        self.is_stationary, self.goal_reached = is_stationary, goal_reached
        self.max_steps, self.max_reobservations = max_steps, max_reobservations
        self.observation_timeout_s = observation_timeout_s
        self.max_observation_age_s = max_observation_age_s
        self.task_timeout_s, self.poll_interval_s = task_timeout_s, poll_interval_s

    def _fresh_stationary(self, after, deadline):
        end = min(deadline, time.monotonic() + self.observation_timeout_s)
        last = after
        stationary_samples = 0
        while time.monotonic() < end:
            obs = self.bridge.adapter.observe()
            obs.validate()
            stamp = _timestamp(obs.timestamp)
            age = datetime.now(timezone.utc).timestamp() - stamp
            if age < 0:
                raise RobotInterfaceError('observation timestamp is in the future')
            if stamp > last and age <= self.max_observation_age_s:
                last = stamp
                if self.is_stationary(obs) is True:
                    stationary_samples += 1
                    if stationary_samples >= 2:
                        return obs
                else:
                    stationary_samples = 0
            time.sleep(min(self.poll_interval_s, max(0, end - time.monotonic())))
        raise RobotInterfaceError('new stationary observation timed out')

    def run(self, task):
        history = []
        deadline = time.monotonic() + self.task_timeout_s
        after = datetime.now(timezone.utc).timestamp()
        retries = 0
        try:
            for _ in range(self.max_steps):
                obs = self._fresh_stationary(after, deadline)
                scenario = observation_to_scenario(obs, task, self.bridge.available_actions)
                if self.goal_reached(obs, deepcopy(task)) is True:
                    return {'status': 'completed', 'history': history}
                decision = self.decide(deepcopy(scenario))
                if time.monotonic() >= deadline:
                    raise RobotInterfaceError('task deadline exceeded during deliberation')
                result = self.bridge.execute_decision(task, decision)
                history.append({'scenario': scenario, 'decision': decision, 'execution': result})
                if result['status'] == 'fail_closed':
                    return {'status': 'fail_closed', 'history': history, 'error': result['error']}
                if result['status'] != 'executed':
                    raise RobotInterfaceError('execution logging failed')
                receipt = result['receipt']
                name = result['approved_action']['name']
                if receipt['status'] == 'failure':
                    raise RobotInterfaceError('action execution failed')
                if name in {'safe_stop', 'ask_human'}:
                    return {'status': 'stopped' if name == 'safe_stop' else 'needs_human',
                            'history': history}
                if name == 'observe_again':
                    if receipt['status'] == 'aborted' and receipt['telemetry'].get('requested_action') != name:
                        raise RobotInterfaceError('unexpected observation abort')
                    retries += 1
                    if retries >= self.max_reobservations:
                        raise RobotInterfaceError('re-observation limit reached')
                else:
                    if receipt['status'] != 'success':
                        raise RobotInterfaceError('motion aborted; cannot resume automatically')
                    retries = 0
                # Accept only frames captured after this action actually returned.
                after = max(_timestamp(receipt['finished_at']),
                            datetime.now(timezone.utc).timestamp())
            raise RobotInterfaceError('task step limit reached')
        except (Exception, KeyboardInterrupt) as exc:
            result = {'status': 'fail_closed', 'error': str(exc), 'history': history}
            try:
                stop = self.bridge.adapter.emergency_stop(str(exc))
                stop.validate()
                result['stop_receipt'] = {'status': stop.status, 'telemetry': stop.telemetry}
            except Exception as stop_exc:
                result['stop_error'] = str(stop_exc)
            return result


def make_memory_decider(store, rule_store, client, *, audit_sink=None):
    """Return a fresh three-Agent + optimizer callback without report caching."""
    from run_three_agents import (ROLE_TOKEN_BUDGET, run_advocate, run_critic,
                                  decide, run_post_deliberation_optimizer)
    from risk_rule_store import resolve_rule_conflicts

    def deliberate(scenario):
        bundles = {role: store.retrieve_memory_cards(
            scenario, role=role, top_k=3, token_budget=ROLE_TOKEN_BUDGET)
            for role in ('advocate', 'critic', 'decision')}
        rules = rule_store.retrieve_matching(scenario)
        resolution = resolve_rule_conflicts(rules, scenario['available_actions'])
        if resolution['status'] == 'conflict_detected':
            raise RobotInterfaceError('conflicting L2 rules require human review')
        advocate, au = run_advocate(scenario, client, bundles['advocate']['items'], rules)
        critic, cu = run_critic(scenario, client, bundles['critic']['items'], rules)
        model, du = decide(scenario, advocate, critic, client, bundles['decision']['items'], rules)
        optimizer, execution = run_post_deliberation_optimizer(
            scenario, bundles['decision']['items'], rules, advocate, critic, model)
        if audit_sink is not None:
            audit_sink(deepcopy({'scenario': scenario, 'memories': bundles, 'rules': rules,
                                'advocate': advocate, 'critic': critic, 'decision': model,
                                'optimizer': optimizer, 'usage': [au, cu, du]}))
        return execution
    return deliberate
