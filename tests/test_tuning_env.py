"""VS_TUNE env override for live A/B of movement knobs (pathfind.tuning_from_env)."""

from __future__ import annotations

from vs_vision_bot.pathfind import Tuning, tuning_from_env


def test_empty_or_missing_is_default():
    assert tuning_from_env(None, log=lambda *_: None) == Tuning()
    assert tuning_from_env("  ", log=lambda *_: None) == Tuning()


def test_overrides_known_fields_and_logs():
    logs: list[str] = []
    t = tuning_from_env("ring_commit_s=2.5, flower_noreverse_s=4;rollout_steps=6", log=logs.append)
    assert t.ring_commit_s == 2.5 and t.flower_noreverse_s == 4.0
    assert t.rollout_steps == 6 and isinstance(t.rollout_steps, int)
    assert t.cover_trigger == Tuning().cover_trigger
    assert any("VS_TUNE overrides" in line and "ring_commit_s=2.5" in line for line in logs)


def test_typos_and_bad_values_are_skipped_not_fatal():
    logs: list[str] = []
    t = tuning_from_env("ring_comit_s=2.5,flower_noreverse_s=abc,ring_commit_s,cover_trigger=0.5", log=logs.append)
    # Bad overrides skipped: untouched fields keep Tuning defaults (now on for flower ring).
    assert t.ring_commit_s == 2.5 and t.flower_noreverse_s == 4.0
    assert t.cover_trigger == 0.5
    assert sum("ignored" in line for line in logs) == 3


def test_main_builds_navigator_from_env():
    import inspect

    from vs_vision_bot import main

    assert 'tuning_from_env(os.environ.get("VS_TUNE"))' in inspect.getsource(main)
