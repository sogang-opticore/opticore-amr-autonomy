"""The bounded learned trial holds zero on faults and never forwards DWA."""
from amr_learned_planner.core.arbiter import ArbiterConfig, CommandArbiter


def test_learned_only_waits_stops_and_recovers():
    arbiter = CommandArbiter(ArbiterConfig(
        mode="learned", fallback_enabled=False, recover_after_sec=0.25))
    assert arbiter.active == "stop"
    assert arbiter.on_cmd("dwa", 1.0, 1.5, 0.1).cmd is None
    assert arbiter.tick(1.0).cmd == (0.0, 0.0)

    arbiter.on_health(1.1, True)
    assert arbiter.on_cmd("learned", 1.1, 0.30, 0.1).cmd is None
    assert arbiter.tick(1.1).cmd == (0.0, 0.0)
    arbiter.on_cmd("learned", 1.4, 0.30, 0.1)
    assert arbiter.tick(1.4).cmd == (0.30, 0.1)
    assert arbiter.active == "learned"

    arbiter.on_health(1.5, False, ["DIAGNOSTIC_HOLD"])
    assert arbiter.on_cmd("learned", 1.5, 0.3, 0.1).cmd == (0.0, 0.0)
    assert arbiter.tick(1.5).cmd == (0.0, 0.0)
    assert arbiter.active == "stop"
    assert arbiter.on_cmd("dwa", 5.0, 1.5, 0.1).cmd is None
    assert arbiter.tick(5.0).cmd == (0.0, 0.0)

    arbiter.on_health(5.1, True)
    arbiter.on_cmd("learned", 5.1, 0.25, 0.0)
    assert arbiter.tick(5.1).cmd == (0.0, 0.0)
    arbiter.on_cmd("learned", 5.4, 0.25, 0.0)
    assert arbiter.tick(5.4).cmd == (0.25, 0.0)
    assert arbiter.active == "learned"
    assert arbiter.tick(6.1).cmd == (0.0, 0.0)
    assert arbiter.active == "stop"


def test_shadow_authority_only_forwards_dwa():
    arbiter = CommandArbiter(ArbiterConfig(mode="shadow", fallback_enabled=False))
    assert arbiter.on_cmd("learned", 1.0, 0.3, 0.1).cmd is None
    assert arbiter.on_cmd("dwa", 1.0, 1.5, 0.0).cmd == (1.5, 0.0)
    arbiter.on_health(1.0, True)
    assert arbiter.tick(1.0).cmd is None
    assert arbiter.active == "dwa"
