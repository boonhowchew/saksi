# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

from datetime import datetime, timedelta, timezone

import pytest

from saksi.workflow import SIGNOFF_QUESTIONS, Project


class TickClock:
    """Deterministic clock: each event is 30 seconds after the previous one."""

    def __init__(self):
        self.t = datetime(2026, 10, 29, 1, 0, tzinfo=timezone.utc)

    def __call__(self):
        self.t += timedelta(seconds=30)
        return self.t.isoformat(timespec="milliseconds")


AUTHOR_A = {"actor_id": "bhc", "name": "Boon-How Chew", "orcid": "0000-0002-8627-6248", "affiliation": "UPM"}
AUTHOR_B = {"actor_id": "coauthor", "name": "Co Author", "affiliation": "UPM"}
REVIEWER = {"actor_id": "rev1", "name": "Re Viewer", "affiliation": "Elsewhere"}
SHA = "a" * 64
ANSWERS = {q: "Checked against the outputs." for q in SIGNOFF_QUESTIONS}
RESTATE = "Booking BMI was associated with higher odds of the outcome in this cohort."


@pytest.fixture
def proj(tmp_path):
    p = Project(tmp_path, "demo", clock=TickClock())
    p.create("Demo study", "primary", AUTHOR_A)
    p.register_human("bhc", AUTHOR_B, "author")
    p.register_human("bhc", REVIEWER, "reviewer")
    return p


def to_analysis(p: Project):
    p.upload_material("bhc", "dataset", "data.csv", SHA, 100)
    p.upload_material("bhc", "protocol", "protocol.pdf", "b" * 64, 200)
    s = p.propose_plan("agent:analyst", [{"id": "primary", "method": "logistic"}, {"id": "sens1", "method": "gbc"}],
                       protocol_ref="doi:10.1136/example")
    for a in ("bhc", "coauthor"):
        s = p.confirm_plan(a, s.plan["sha"])
    return s


def run_planned(p: Project):
    for pid in ("primary", "sens1"):
        p.complete_analysis("agent:analyst", pid, plan_analysis_id=pid, spec={"id": pid}, code_sha256=SHA,
                            outputs_sha256="c" * 64, execution={"env": {}}, status="ok")
    return p.state()


def to_revision(p: Project):
    to_analysis(p)
    s = run_planned(p)
    for a in ("bhc", "coauthor"):
        s = p.accept_analyses(a, s.analysis_set_sha())
    return p.create_draft("agent:writer", "Draft one text about results.", ["Methods", "Results"],
                          ["primary", "sens1"], call=None)


def sign_all(p: Project):
    s = p.state()
    for a in s.authors:
        if a in s.signoffs.get(s.draft_version, {}):
            continue
        p.open_draft(a, s.draft_version)
        s = p.sign_off(a, s.draft_version, RESTATE, ANSWERS)
    return s
