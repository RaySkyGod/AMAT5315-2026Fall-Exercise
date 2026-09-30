"""Pytest for the checkpoint audit logic on synthetic action tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_check import audit_actions


def real_n2_delta1():
    # schedule(2, 1) as the Rust scheduler emits it: s_0 kept, grads 1 then 0
    return [
        {"action": "restore", "n": 0, "saved_states": 1},
        {"action": "call", "n": 0, "saved_states": 1},
        {"action": "store", "n": 1, "saved_states": 2},
        {"action": "grad", "n": 1, "saved_states": 2},
        {"action": "fetch", "n": 1, "saved_states": 1},
        {"action": "grad", "n": 0, "saved_states": 1},
    ]


def test_good_schedule_passes():
    g, r, b = audit_actions(real_n2_delta1(), n_steps=2, delta=1)
    assert (g, r, b) == (0, 0, 0)


def test_detects_grad_order_violation():
    acts = real_n2_delta1()
    # swap the two grads: order 0 then 1 instead of 1 then 0
    acts = [a for a in acts if a["action"] != "grad"][:3] + [
        {"action": "grad", "n": 0, "saved_states": 2},
        {"action": "fetch", "n": 1, "saved_states": 1},
        {"action": "grad", "n": 1, "saved_states": 1},   # slot 1 already fetched
    ]
    g, r, b = audit_actions(acts, n_steps=2, delta=1)
    assert g > 0          # order 0,1 mismatches expected 1,0
    assert r == 1         # grad on the fetched slot
    assert b == 0


def test_detects_missing_grad():
    acts = [a for a in real_n2_delta1() if not (a["action"] == "grad" and a["n"] == 0)]
    g, _, _ = audit_actions(acts, n_steps=2, delta=1)
    assert g > 0          # grad 0 missing


def test_detects_illegal_restore_and_overbudget():
    acts = [
        {"action": "restore", "n": 5, "saved_states": 1},   # never stored
        {"action": "store", "n": 3, "saved_states": 2},
        {"action": "grad", "n": 3, "saved_states": 3},      # wrong count -> over budget
        {"action": "grad", "n": 0, "saved_states": 3},
    ]
    g, r, b = audit_actions(acts, n_steps=4, delta=1)
    assert r == 1
    assert b >= 2


def test_detects_s0_not_left():
    acts = real_n2_delta1() + [{"action": "fetch", "n": 0, "saved_states": 0}]
    _, _, b = audit_actions(acts, n_steps=2, delta=1)
    assert b > 0          # end state must be exactly {0}
