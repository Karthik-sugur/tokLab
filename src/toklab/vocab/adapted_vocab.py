"""Append-only adapted vocabulary with content hashing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toklab.core.hashing import config_hash


@dataclass(frozen=True)
class AdaptedEntry:
    token: str
    id: int
    gain: float
    freq: int
    n_repos: int


class AdaptedVocab:
    """Custom tokens with IDs starting at ``id_offset`` (== base vocab size)."""

    def __init__(
        self,
        *,
        id_offset: int,
        entries: list[AdaptedEntry] | None = None,
        mode: str = "native",
        base_model: str = "",
    ) -> None:
        if id_offset < 0:
            raise ValueError("id_offset must be >= 0")
        self.id_offset = id_offset
        self.mode = mode
        self.base_model = base_model
        self._entries: list[AdaptedEntry] = []
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        for e in entries or []:
            self._add_entry(e)

    def _add_entry(self, entry: AdaptedEntry) -> None:
        if entry.id < self.id_offset:
            raise ValueError(
                f"adapted id {entry.id} < id_offset {self.id_offset} "
                "(append-only after base vocab)"
            )
        if entry.token in self._token_to_id:
            raise ValueError(f"duplicate adapted token {entry.token!r}")
        if entry.id in self._id_to_token:
            raise ValueError(f"duplicate adapted id {entry.id}")
        self._entries.append(entry)
        self._token_to_id[entry.token] = entry.id
        self._id_to_token[entry.id] = entry.token

    @classmethod
    def from_selected(
        cls,
        selected: list[Any],
        *,
        id_offset: int,
        mode: str,
        base_model: str = "",
    ) -> AdaptedVocab:
        """Build vocab from scored candidates; IDs are id_offset, id_offset+1, …"""
        entries: list[AdaptedEntry] = []
        for i, s in enumerate(selected):
            entries.append(
                AdaptedEntry(
                    token=s.token,
                    id=id_offset + i,
                    gain=float(s.gain),
                    freq=int(s.freq),
                    n_repos=int(s.n_repos),
                )
            )
        return cls(
            id_offset=id_offset,
            entries=entries,
            mode=mode,
            base_model=base_model,
        )

    @property
    def entries(self) -> tuple[AdaptedEntry, ...]:
        return tuple(self._entries)

    def as_dict(self) -> dict[str, int]:
        """token → id map for longest-match lookup."""
        return dict(self._token_to_id)

    def reverse_dict(self) -> dict[int, str]:
        return dict(self._id_to_token)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def content_hash(self) -> str:
        payload = {
            "id_offset": self.id_offset,
            "mode": self.mode,
            "base_model": self.base_model,
            "entries": [
                {
                    "token": e.token,
                    "id": e.id,
                    "gain": e.gain,
                    "freq": e.freq,
                    "n_repos": e.n_repos,
                }
                for e in self._entries
            ],
        }
        return config_hash(payload)

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "id_offset": self.id_offset,
            "mode": self.mode,
            "base_model": self.base_model,
            "content_hash": self.content_hash(),
            "entries": [
                {
                    "token": e.token,
                    "id": e.id,
                    "gain": e.gain,
                    "freq": e.freq,
                    "n_repos": e.n_repos,
                }
                for e in self._entries
            ],
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json_obj(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> AdaptedVocab:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = [
            AdaptedEntry(
                token=e["token"],
                id=int(e["id"]),
                gain=float(e["gain"]),
                freq=int(e["freq"]),
                n_repos=int(e["n_repos"]),
            )
            for e in data["entries"]
        ]
        vocab = cls(
            id_offset=int(data["id_offset"]),
            entries=entries,
            mode=str(data.get("mode", "")),
            base_model=str(data.get("base_model", "")),
        )
        stored = data.get("content_hash")
        if stored is not None and stored != vocab.content_hash():
            raise ValueError(
                f"content_hash mismatch for {path}: "
                f"stored={stored} computed={vocab.content_hash()}"
            )
        return vocab


def empty_adapted_vocab(
    id_offset: int, *, mode: str, base_model: str = ""
) -> AdaptedVocab:
    """K=0 vocabulary (no entries)."""
    return AdaptedVocab(
        id_offset=id_offset, entries=[], mode=mode, base_model=base_model
    )
