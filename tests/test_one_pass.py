"""Covers the drift between what ``build_agent`` builds and what ``print_caveat`` claims.

``2_one_pass.py`` can't be imported with a plain ``import`` statement (the
filename starts with a digit), and its module body must not run the actual
demo on import - it's loaded by file path, with ``main()`` guarded so only
running it as a script executes anything expensive.
"""
import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parent.parent / "2_one_pass.py"
_SPEC = importlib.util.spec_from_file_location("one_pass_demo", _PATH)
one_pass = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(one_pass)


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_blank_or_unset_checkpoint_env_resolves_to_none(raw):
    assert one_pass.resolve_checkpoint(raw) is None


def test_nonblank_checkpoint_that_is_not_a_real_directory_exits_loudly(tmp_path):
    with pytest.raises(SystemExit, match=str(tmp_path)):
        one_pass.resolve_checkpoint(str(tmp_path))  # empty dir: no encoder/, no rl_agent_config.json


def test_empty_string_checkpoint_builds_the_untrained_mode(monkeypatch):
    # An exported-but-empty OPENJEV_CHECKPOINT must take the same path as unset.
    monkeypatch.setattr(one_pass.openjev, "Agent", lambda: "stub-agent")
    agent, mode = one_pass.build_agent(tiny=False, checkpoint_path=one_pass.resolve_checkpoint(""))
    assert mode.kind == "untrained"
    assert agent == "stub-agent"


def test_untrained_mode_caveat_never_claims_the_real_comparison(capsys):
    one_pass.print_caveat(one_pass.SimpleNamespace(kind="untrained"))
    printed = capsys.readouterr().out
    assert "real comparison" not in printed
    assert "untrained" in printed


def test_empty_env_var_end_to_end_does_not_announce_a_trained_checkpoint(monkeypatch, capsys):
    """The regression itself: OPENJEV_CHECKPOINT="" must not reach the "this is the real
    comparison" line, which is what happened when build_agent tested truthiness while
    print_caveat tested identity-to-None on two different copies of the same value."""
    monkeypatch.setattr(one_pass.openjev, "Agent", lambda: "stub-agent")
    checkpoint_path = one_pass.resolve_checkpoint("")
    _, mode = one_pass.build_agent(tiny=False, checkpoint_path=checkpoint_path)
    one_pass.print_caveat(mode)
    printed = capsys.readouterr().out
    assert "real comparison" not in printed
    assert "untrained" in printed


def test_checkpoint_mode_caveat_echoes_the_resolved_path(capsys):
    one_pass.print_caveat(one_pass.SimpleNamespace(kind="checkpoint", path="/models/run-7"))
    printed = capsys.readouterr().out
    assert "/models/run-7" in printed
    assert "real comparison" in printed
