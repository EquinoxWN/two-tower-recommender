"""The twotower command."""

from __future__ import annotations

from pathlib import Path

import pytest

from two_tower_recommender.cli import main


def test_the_synthetic_run_prints_metrics_and_latency(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--synthetic", "--epochs", "2"]) == 0
    out = capsys.readouterr().out
    assert "| two-tower + FAISS (exact) |" in out
    assert "| popularity |" in out
    assert "p99" in out


def test_a_bad_archive_exits_with_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "ml.zip"
    bad.write_bytes(b"junk")
    monkeypatch.setattr("two_tower_recommender.cli.download", lambda path: path)
    assert main(["--data", str(bad)]) == 2
    assert "not a zip" in capsys.readouterr().err


@pytest.mark.parametrize("epochs", ["0", "-1", "1001"])
def test_an_epoch_count_out_of_range_is_refused(epochs: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--synthetic", "--epochs", epochs])
    assert exc.value.code == 2
