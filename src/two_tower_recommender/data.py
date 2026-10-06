"""MovieLens 1M loading (checksum-verified download), a time-based split, and a synthetic dataset
with planted preferences for tests."""

from __future__ import annotations

import hashlib
import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
# GroupLens publishes the MD5 (c4d9eecfca2ab87c1945afe126590906); the SHA-256 below was taken from
# a download whose MD5 matched it.
ML_1M_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
AGE_BUCKETS = [1, 18, 25, 35, 45, 50, 56]
POSITIVE_RATING = 4


class DataError(ValueError):
    """The dataset is missing, corrupt or malformed."""


@dataclass
class Dataset:
    """Users, items and timestamped interactions, with contiguous integer ids."""

    n_users: int
    n_items: int
    user_gender: np.ndarray  # (n_users,) 0 or 1
    user_age: np.ndarray  # (n_users,) bucket index
    user_occupation: np.ndarray  # (n_users,) 0..20
    item_genres: np.ndarray  # (n_items, n_genres) multi-hot float32
    item_year: np.ndarray  # (n_items,) decade bucket index
    genres: list[str]
    titles: list[str]
    users: np.ndarray  # interactions: user index
    items: np.ndarray  # interactions: item index
    ratings: np.ndarray
    times: np.ndarray


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path, url: str = ML_1M_URL, expected_sha256: str = ML_1M_SHA256) -> Path:
    """Download the archive once and refuse it unless its checksum matches."""
    if dest.exists() and sha256(dest) == expected_sha256:
        return dest
    if not url.startswith("https://"):
        raise DataError(f"refusing to download over anything but https: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:  # noqa: S310 - https only
        size = 0
        while chunk := r.read(1 << 20):
            size += len(chunk)
            if size > MAX_DOWNLOAD_BYTES:
                f.close()
                tmp.unlink()
                raise DataError(f"download larger than {MAX_DOWNLOAD_BYTES} bytes")
            f.write(chunk)
    got = sha256(tmp)
    if got != expected_sha256:
        tmp.unlink()
        raise DataError(f"checksum mismatch for {url}: got {got}")
    tmp.replace(dest)
    return dest


def _lines(z: zipfile.ZipFile, name: str) -> list[list[str]]:
    try:
        raw = z.read(f"ml-1m/{name}")
    except KeyError as e:
        raise DataError(f"archive has no ml-1m/{name}") from e
    text = io.TextIOWrapper(io.BytesIO(raw), encoding="latin-1").read()
    return [line.split("::") for line in text.splitlines() if line]


def load_movielens(path: Path) -> Dataset:
    """Parse the ml-1m archive (users, movies with genres and year, ratings with timestamps)."""
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise DataError(f"{path} is not a zip archive") from e
    with z:
        users = _lines(z, "users.dat")
        movies = _lines(z, "movies.dat")
        ratings = _lines(z, "ratings.dat")
    uid = {int(u[0]): i for i, u in enumerate(users)}
    occ = np.array([int(u[3]) for u in users], dtype=np.int64)
    gender = np.array([1 if u[1] == "F" else 0 for u in users], dtype=np.int64)
    age = np.array([AGE_BUCKETS.index(int(u[2])) for u in users], dtype=np.int64)
    genre_names = sorted({g for m in movies for g in m[2].split("|")})
    gidx = {g: i for i, g in enumerate(genre_names)}
    mid = {int(m[0]): i for i, m in enumerate(movies)}
    multi = np.zeros((len(movies), len(genre_names)), dtype=np.float32)
    years = np.zeros(len(movies), dtype=np.int64)
    titles = []
    for i, m in enumerate(movies):
        for g in m[2].split("|"):
            multi[i, gidx[g]] = 1.0
        found = re.search(r"\((\d{4})\)\s*$", m[1])
        year = int(found.group(1)) if found else 1990
        years[i] = min(max((year - 1910) // 10, 0), 9)
        titles.append(m[1])
    r = np.array(
        [[uid[int(x[0])], mid[int(x[1])], int(x[2]), int(x[3])] for x in ratings], dtype=np.int64
    )
    return Dataset(
        n_users=len(users),
        n_items=len(movies),
        user_gender=gender,
        user_age=age,
        user_occupation=occ,
        item_genres=multi,
        item_year=years,
        genres=genre_names,
        titles=titles,
        users=r[:, 0],
        items=r[:, 1],
        ratings=r[:, 2],
        times=r[:, 3],
    )


@dataclass
class Split:
    """Training positives, and each test user's held-out later positives."""

    train_users: np.ndarray
    train_items: np.ndarray
    test: dict[int, list[int]]
    seen: dict[int, set[int]]  # every item each user rated before the test period


def split_by_time(ds: Dataset, holdout: int = 2, min_train: int = 5) -> Split:
    """Per user, the last `holdout` positives (rating >= 4) by time are the test set.

    Everything the user rated before them, positive or not, is "seen" and excluded from that
    user's recommendations, so the model is never rewarded for recommending what was already
    watched.
    """
    order = np.lexsort((ds.items, ds.times, ds.users))
    u, i, r = ds.users[order], ds.items[order], ds.ratings[order]
    train_u: list[int] = []
    train_i: list[int] = []
    test: dict[int, list[int]] = {}
    seen: dict[int, set[int]] = {}
    bounds = np.flatnonzero(np.diff(u)) + 1
    for start, end in zip(np.r_[0, bounds], np.r_[bounds, len(u)], strict=True):
        user = int(u[start])
        pos = [k for k in range(start, end) if r[k] >= POSITIVE_RATING]
        if len(pos) < holdout + min_train:
            for k in pos:
                train_u.append(user)
                train_i.append(int(i[k]))
            seen[user] = {int(x) for x in i[start:end]}
            continue
        cut = pos[-holdout]
        test[user] = [int(i[k]) for k in pos[-holdout:]]
        seen[user] = {int(x) for x in i[start:cut]}
        for k in pos[:-holdout]:
            train_u.append(user)
            train_i.append(int(i[k]))
    return Split(np.array(train_u, dtype=np.int64), np.array(train_i, dtype=np.int64), test, seen)


def synthetic(
    n_users: int = 600, n_items: int = 400, n_genres: int = 8, per_user: int = 40, seed: int = 0
) -> Dataset:
    """A dataset where each user likes two genres and picks items from them (with popular items
    picked more often), so a personalized model can beat popularity by a clear margin."""
    rng = np.random.default_rng(seed)
    item_genre = rng.integers(0, n_genres, n_items)
    multi = np.zeros((n_items, n_genres), dtype=np.float32)
    multi[np.arange(n_items), item_genre] = 1.0
    popularity = rng.zipf(1.6, n_items).astype(float)
    likes = np.stack([rng.choice(n_genres, 2, replace=False) for _ in range(n_users)])
    users, items, times = [], [], []
    for u in range(n_users):
        pool = np.flatnonzero(np.isin(item_genre, likes[u]))
        p = popularity[pool] / popularity[pool].sum()
        chosen = rng.choice(pool, size=min(per_user, len(pool)), replace=False, p=p)
        users += [u] * len(chosen)
        items += chosen.tolist()
        times += sorted(rng.integers(0, 10**6, len(chosen)).tolist())
    n = len(users)
    return Dataset(
        n_users=n_users,
        n_items=n_items,
        user_gender=rng.integers(0, 2, n_users),
        user_age=rng.integers(0, len(AGE_BUCKETS), n_users),
        user_occupation=rng.integers(0, 21, n_users),
        item_genres=multi,
        item_year=rng.integers(0, 10, n_items),
        genres=[f"genre{g}" for g in range(n_genres)],
        titles=[f"item {i}" for i in range(n_items)],
        users=np.array(users, dtype=np.int64),
        items=np.array(items, dtype=np.int64),
        ratings=np.full(n, 5, dtype=np.int64),
        times=np.array(times, dtype=np.int64),
    )
