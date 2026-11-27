from types import SimpleNamespace

import pytest

from run_fleet import (DRIVER, MANAGER, MISSIONS, GoalLedger, ProtocolFailure, check_distance, endpoint_gids,
                       physical_stop)


@pytest.fixture
def ledger():
    writers = {r: {DRIVER: r + '-driver', MANAGER: r + '-manager'} for r in MISSIONS}
    result = GoalLedger(writers, (2., 22.))
    result.issued = {'amr2': 'map:2.000000000', 'amr1': 'map:6.000000000'}
    return result


def sample(second, xy, writer='amr2-manager', robot='amr2'):
    return {'robot': robot, 'writer': writer, 'is_map': True, 'sec': second, 'nsec': 0, 'xy': list(xy),
            'orientation': [0., 0., 0., 1.]}


def test_attribution_and_restored_goal_require_original_destination(ledger):
    assert ledger.observe(sample(2, MISSIONS['amr2'], 'amr2-driver')) == ('mission', 'map:2.000000000')
    assert ledger.observe(sample(3, [2.02, 22.01])) == ('hold', 'map:3.000000000')
    assert ledger.observe(sample(4, [2.02, 22.01])) == ('hold', 'map:4.000000000')
    assert ledger.observe(sample(6, MISSIONS['amr1'], 'amr1-driver', 'amr1'))[0] == 'mission'
    assert ledger.observe(sample(7, MISSIONS['amr2']))[0] == 'resume'
    with pytest.raises(ProtocolFailure):
        ledger.observe(sample(8, [2.02, 22.01]))


@pytest.mark.parametrize('message', [sample(3, [2., 22.], 'amr2-driver'), sample(3, MISSIONS['amr2']),
                                     sample(3, [2., 22.], 'amr1-manager', 'amr1'),
                                     {**sample(3, [2., 22.]), 'is_map': False}])
def test_cannot_count_probe_hold_wrong_robot_or_unrestored_mission(ledger, message):
    with pytest.raises(ProtocolFailure):
        ledger.observe(message)


def test_foreign_gid_is_an_invalid_endpoint_and_late_reused_stamp_is_a_failure(ledger):
    with pytest.raises(RuntimeError, match='unregistered DDS'):
        ledger.observe(sample(3, [2., 22.], 'amr1-manager'))
    event = sample(3, [2., 22.])
    ledger.observe(event)
    with pytest.raises(ProtocolFailure, match='reused'):
        ledger.observe(event)


def test_gid_snapshot_keeps_the_entire_identifier_and_rejects_duplicate_and_empty():
    def capture(ids):
        pub = [SimpleNamespace(node_namespace='/', node_name='deadlock_manager', endpoint_gid=i) for i in ids]
        return endpoint_gids(SimpleNamespace(get_publishers_info_by_topic=lambda topic: pub), '/amr2/goal_pose')
    assert capture([list(range(24))]) == {MANAGER: bytes(range(24)).hex()}
    with pytest.raises(RuntimeError):
        capture([[0] * 24])
    with pytest.raises(RuntimeError):
        capture([list(range(24)), list(range(1, 25))])


def test_stop_boundary_ends_approach_before_contact_and_stop_needs_physics():
    assert check_distance({'amr1': [3.6, 22], 'amr2': [2, 22]}) == pytest.approx(1.6)
    with pytest.raises(ProtocolFailure):
        check_distance({'amr1': [3.6, 22], 'amr2': [2.45, 22]})
    with pytest.raises(RuntimeError, match='nonfinite'):
        check_distance({'amr1': [float('nan'), 22], 'amr2': [2., 22]})
    assert physical_stop({'cmd': (0., 0.), 'motion': (2., 22., .005, .01)})
    assert not physical_stop({'cmd': (0., 0.), 'motion': (2., 22., .05, .01)})
    assert not physical_stop({'cmd': (0., .03), 'motion': (2., 22., .005, .01)})
