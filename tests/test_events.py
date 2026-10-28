# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

import json

from saksi.events import EventLog


def make(tmp_path, n=5):
    log = EventLog(tmp_path / "e.jsonl", "p1")
    for i in range(n):
        log.append("bhc", "note", {"i": i})
    return log


def rewrite(log, fn):
    lines = log.path.read_text().splitlines()
    lines = fn(lines)
    log.path.write_text("\n".join(lines) + "\n")


def test_chain_verifies(tmp_path):
    log = make(tmp_path)
    r = log.verify()
    assert r.ok and r.events == 5 and r.head == log.events()[-1].hash


def test_detects_edited_payload(tmp_path):
    log = make(tmp_path)

    def tamper(lines):
        ev = json.loads(lines[2])
        ev["payload"]["i"] = 99
        lines[2] = json.dumps(ev)
        return lines

    rewrite(log, tamper)
    r = log.verify()
    assert not r.ok and r.bad_seq == 3 and "altered" in r.error


def test_detects_rehashed_edit(tmp_path):
    """Editing an event and recomputing its own hash still breaks the next link."""
    log = make(tmp_path)
    evs = log.events()
    from dataclasses import asdict
    from saksi.events import Event
    bad = Event(**{**asdict(evs[2]), "payload": {"i": 99}, "hash": ""})
    bad = Event(**{**asdict(bad), "hash": bad.compute_hash()})

    def tamper(lines):
        lines[2] = bad.to_json()
        return lines

    rewrite(log, tamper)
    r = log.verify()
    assert not r.ok and r.bad_seq == 4 and "chain broken" in r.error


def test_detects_deletion_and_reorder(tmp_path):
    log = make(tmp_path)
    rewrite(log, lambda lines: lines[:2] + lines[3:])
    assert not log.verify().ok
    log2 = make(tmp_path / "b")
    rewrite(log2, lambda lines: [lines[1], lines[0]] + lines[2:])
    assert not log2.verify().ok


def test_detects_truncated_line(tmp_path):
    log = make(tmp_path)
    rewrite(log, lambda lines: lines[:-1] + [lines[-1][:20]])
    r = log.verify()
    assert not r.ok and "unreadable" in r.error
