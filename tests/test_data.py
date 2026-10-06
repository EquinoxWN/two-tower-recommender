"""Loading, the time split and the verified download."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from two_tower_recommender import data
from two_tower_recommender.data import DataError, download, load_movielens, split_by_time, synthetic


def tiny_archive(path: Path) -> Path:
    """A MovieLens-format archive with 3 users, 4 movies and 14 ratings."""
    files = {
        "ml-1m/users.dat": "1::F::1::10::48067\n2::M::56::16::70072\n3::M::25::15::55117\n",
        "ml-1m/movies.dat": "10::Toy Story (1995)::Animation|Children's|Comedy\n20::Heat (1995)::Action|Crime\n"
        "30::Old Movie (1931)::Drama\n40::No Year::Comedy\n",
        "ml-1m/ratings.dat": "".join(
            f"{u}::{m}::{r}::{t}\n"
            for u, m, r, t in [
                (1, 10, 5, 100),
                (1, 20, 4, 200),
                (1, 30, 5, 300),
                (1, 40, 4, 400),
                (1, 10, 2, 50),
                (2, 20, 5, 10),
                (2, 30, 1, 20),
                (3, 10, 4, 1),
                (3, 20, 4, 2),
                (3, 30, 4, 3),
                (3, 40, 5, 4),
                (3, 10, 4, 5),
                (3, 20, 4, 6),
                (3, 30, 3, 7),
            ]
        ),
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return path


def test_movielens_files_are_parsed_and_ids_made_contiguous(tmp_path: Path) -> None:
    ds = load_movielens(tiny_archive(tmp_path / "ml.zip"))
    assert (ds.n_users, ds.n_items, len(ds.users)) == (3, 4, 14)
    assert ds.user_gender.tolist() == [1, 0, 0]
    assert ds.user_age.tolist() == [0, 6, 2]
    assert ds.genres == ["Action", "Animation", "Children's", "Comedy", "Crime", "Drama"]
    assert ds.item_genres[0].tolist() == [0, 1, 1, 1, 0, 0]
    assert ds.item_year.tolist() == [8, 8, 2, 8], (
        "1995 -> 1990s bucket, 1931 -> 1930s, no year -> 1990"
    )
    assert set(ds.items.tolist()) == {0, 1, 2, 3}


def test_the_split_holds_out_each_users_last_positives(tmp_path: Path) -> None:
    ds = load_movielens(tiny_archive(tmp_path / "ml.zip"))
    sp = split_by_time(ds, holdout=2, min_train=2)
    # User 1 (index 0): positives at times 100..400, the last two (items 30 and 40) are held out.
    # User 3 (index 2): positives at times 1..6, the last two (items 10 and 20 again) are held out.
    assert sp.test == {0: [2, 3], 2: [0, 1]}
    assert 1 not in sp.test, "user 2 has too few positives to hold any out"
    assert sp.seen[2] == {0, 1, 2, 3}, "everything user 3 rated before the test period is seen"
    for user, held in sp.test.items():
        train_times = ds.times[(ds.users == user) & np.isin(ds.items, list(sp.seen[user]))]
        assert train_times.max() <= ds.times[ds.users == user].max()
        assert held
    # Ratings below 4 are not positives; user 1's rating of 2 at time 50 never becomes training data.
    pairs = set(zip(sp.train_users.tolist(), sp.train_items.tolist(), strict=True))
    assert (1, 2) not in pairs, "user 2 rated item 30 with 1 star"


def test_no_test_interaction_is_older_than_the_users_training_data() -> None:
    ds = synthetic(n_users=50, n_items=80, seed=3)
    sp = split_by_time(ds)
    first_test: dict[int, int] = {}
    for u, i, t in zip(ds.users.tolist(), ds.items.tolist(), ds.times.tolist(), strict=True):
        if u in sp.test and i in sp.test[u]:
            first_test[u] = min(first_test.get(u, t), t)
    for u, i in zip(sp.train_users.tolist(), sp.train_items.tolist(), strict=True):
        if u in first_test:
            t = ds.times[(ds.users == u) & (ds.items == i)].max()
            assert t <= first_test[u], "training data must come before the held-out positives"


def test_a_corrupt_or_wrong_archive_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(DataError, match="not a zip"):
        load_movielens(bad)
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("other.txt", "x")
    with pytest.raises(DataError, match=r"no ml-1m/users\.dat"):
        load_movielens(empty)


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_downloads_are_kept_only_when_the_checksum_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"archive bytes"
    good = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        "two_tower_recommender.data.urllib.request.urlopen",
        lambda url, timeout: FakeResponse(payload),
    )
    dest = tmp_path / "d" / "ml.zip"
    assert download(dest, "https://example.test/ml.zip", good) == dest
    assert dest.read_bytes() == payload
    other = tmp_path / "other.zip"
    with pytest.raises(DataError, match="checksum mismatch"):
        download(other, "https://example.test/ml.zip", "0" * 64)
    assert not other.exists()
    assert not other.with_suffix(".part").exists(), "a rejected download leaves nothing behind"
    with pytest.raises(DataError, match="https"):
        download(tmp_path / "x.zip", "http://example.test/ml.zip", good)


def test_oversized_downloads_are_cut_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(
        "two_tower_recommender.data.urllib.request.urlopen",
        lambda url, timeout: FakeResponse(b"x" * 100),
    )
    with pytest.raises(DataError, match="larger than"):
        download(tmp_path / "big.zip", "https://example.test/big.zip", "0" * 64)
    assert not (tmp_path / "big.part").exists()


def test_synthetic_data_is_seeded() -> None:
    a, b, c = synthetic(seed=1), synthetic(seed=1), synthetic(seed=2)
    assert np.array_equal(a.items, b.items)
    assert not np.array_equal(a.items, c.items)
