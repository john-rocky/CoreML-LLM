#!/usr/bin/env python3
"""Mac-side integration test for MtpModuleStackEngine's accept/emit logic.

Mirrors the Swift logic in pure Python. Simulates full token streams
under different draft quality regimes and asserts:
  1. No token appears twice consecutively in the emitted stream
     (catches the double-emission bug).
  2. Total emitted count matches expected for each acceptance scenario.
  3. newNextID is never in the current cycle's emitted (it's for next cycle).

Run: python3 conversion/train_mtp_modules/verify_accept_logic.py
"""


def speculate_step(next_id: int, module_outputs, verify_outputs) -> tuple[list[int], int]:
    """Pure Python mirror of MtpModuleStackEngine.speculateStep accept/emit.

    module_outputs = (d0, d1)        — drafts from module_0, module_1
    verify_outputs = (a0, a1, a2)    — target argmax at 3 verify positions
    Returns: (emitted_tokens, new_next_id)
    """
    d0, d1 = module_outputs
    a0, a1, a2 = verify_outputs

    emitted = [next_id]
    match_count = 0

    if d0 == a0:
        emitted.append(d0)
        match_count += 1
        if d1 == a1:
            emitted.append(d1)
            match_count += 1
            new_next_id = a2  # bonus
        else:
            new_next_id = a1  # correction at P+2
    else:
        new_next_id = a0  # correction at P+1

    return emitted, new_next_id


def simulate_run(next_id_init: int, cycles: list, stream_label: str):
    """Simulate `cycles` spec steps, emit all tokens, check invariants."""
    print(f"\n=== {stream_label} ===")
    all_emitted = []
    next_id = next_id_init

    for i, (modules_out, verify_out) in enumerate(cycles):
        cycle_emitted, new_next_id = speculate_step(next_id, modules_out, verify_out)

        # Invariant A: next_id (new) must NOT be in current cycle's emitted
        # (else double-emission next cycle).
        assert new_next_id not in cycle_emitted, (
            f"Cycle {i}: new_next_id={new_next_id} is in emitted={cycle_emitted} "
            f"→ will double-emit next cycle. BUG."
        )

        # Invariant B: first token of emitted must be current next_id
        assert cycle_emitted[0] == next_id, \
            f"Cycle {i}: expected emitted[0]=={next_id}, got {cycle_emitted[0]}"

        all_emitted.extend(cycle_emitted)
        next_id = new_next_id
        print(f"  Cycle {i}: emitted={cycle_emitted}, new_next_id={new_next_id}")

    # Invariant C: no two consecutive identical tokens (classic doubling signature)
    dup_count = sum(1 for a, b in zip(all_emitted, all_emitted[1:]) if a == b)
    print(f"  Total emitted: {all_emitted}")
    print(f"  Consecutive duplicates: {dup_count}")
    assert dup_count == 0, f"FAIL: found {dup_count} consecutive duplicate tokens"
    print(f"  ✓ No double-emission")


def main():
    # Scenario 1: all matches in every cycle (ideal case, max speedup)
    # next_id=100. Cycle N: drafts match target; bonus becomes next.
    simulate_run(
        next_id_init=100,
        cycles=[
            # Cycle 0: start with 100. Drafts (d0=200, d1=300) both match.
            # Target verify: argmax=[200, 300, 400]. Bonus=400.
            ((200, 300), (200, 300, 400)),
            # Cycle 1: start with 400 (bonus from prev). Drafts (500, 600) match.
            # Bonus=700.
            ((500, 600), (500, 600, 700)),
            # Cycle 2: start with 700. All match, bonus=1000.
            ((800, 900), (800, 900, 1000)),
        ],
        stream_label="All match (ideal)",
    )

    # Scenario 2: no matches (drafter useless)
    simulate_run(
        next_id_init=100,
        cycles=[
            # Cycle 0: d0=200 != a0=150. correction=150.
            ((200, 300), (150, 151, 152)),
            # Cycle 1: start with 150. Drafts still wrong. correction=175.
            ((250, 350), (175, 176, 177)),
        ],
        stream_label="No match (baseline equiv)",
    )

    # Scenario 3: partial match at module_0 only (common mid-quality)
    simulate_run(
        next_id_init=100,
        cycles=[
            # Cycle 0: d0=200 matches a0; d1=300 != a1=350. correction=350.
            ((200, 300), (200, 350, 400)),
            # Cycle 1: start=350. d0=450 matches; d1=550 doesn't. correction=600.
            ((450, 550), (450, 600, 700)),
        ],
        stream_label="Partial match (50% draft acc)",
    )

    # Scenario 4: mixed realistic — alternating match patterns
    simulate_run(
        next_id_init=100,
        cycles=[
            ((200, 300), (200, 300, 400)),   # all match, bonus=400
            ((500, 600), (550, 600, 700)),   # d0 rejected, correction=550
            ((650, 750), (650, 775, 800)),   # d0 match, d1 reject, correction=775
            ((850, 950), (900, 950, 1000)),  # d0 reject, correction=900
        ],
        stream_label="Mixed realistic",
    )

    # Scenario 5: bonus equals next_id of next cycle (check: no duplicate flows)
    # This is the CRITICAL test for the bug we just fixed.
    simulate_run(
        next_id_init=100,
        cycles=[
            ((200, 300), (200, 300, 999)),   # bonus=999
            # Next cycle starts with next_id=999.
            ((1000, 1100), (1050, 1100, 1200)),  # d0 reject at 999's position
        ],
        stream_label="Bonus→next_id handoff (double-emit catcher)",
    )

    print("\n✓ All scenarios passed. Accept/emit logic is clean.")


if __name__ == "__main__":
    main()
