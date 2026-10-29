# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Deploy Saksi on Modal.

  modal secret create saksi-keys META_API_KEY=... BASETEN_API_KEY=... SAKSI_CREATE_TOKEN=...
  modal run modal_app.py::smoke      # check both Meta model endpoints
  modal run modal_app.py::demo       # live end-to-end run on synthetic data
  modal deploy modal_app.py          # serve the API

Study materials and the event logs live on a private Modal Volume. API keys
live in a Modal Secret and never reach the browser. Analyses run in separate
Modal Sandboxes with networking blocked.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("saksi")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi==0.162.0", "httpx==0.28.1", "pydantic==2.13.5", "python-multipart",
        "numpy==2.5.4", "pandas==3.1.0", "scipy==1.18.1", "statsmodels==0.15.0", "scikit-learn==1.9.1",
        "openpyxl", "pypdf", "python-docx",
    )
    .env({"SAKSI_WEB_DIR": "/web"})
    .add_local_python_source("saksi")
    # Modal ships only .py files with the package, so the web page goes separately.
    .add_local_dir(Path(__file__).parent / "saksi" / "static", remote_path="/web")
)

volume = modal.Volume.from_name("saksi-data", create_if_missing=True)
keys = modal.Secret.from_name("saksi-keys")
DATA = "/data"


def _agents_factory(project):
    from saksi.agents import Agents
    from saksi.llm import ModelClient
    from saksi.sandbox import ModalExecutor
    return Agents(project, ModelClient(project.blobs), ModalExecutor())


@app.function(image=image, volumes={DATA: volume}, secrets=[keys],
              max_containers=1,          # one writer per event log
              timeout=900)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from saksi.api import create_app
    return create_app(f"{DATA}/projects", on_write=volume.commit, agents_factory=_agents_factory)


@app.function(image=image, secrets=[keys], timeout=300)
def smoke() -> list[dict]:
    """One tiny call to each Meta model endpoint. Prints model, tokens and latency only."""
    import tempfile

    from saksi.blobs import BlobStore
    from saksi.llm import ModelClient

    client = ModelClient(BlobStore(tempfile.mkdtemp()), mode="live")
    msg = [{"role": "user", "content": "Reply with exactly: saksi ok"}]
    results = []
    for provider in ("meta", "baseten"):
        try:
            # Generous cap: Muse Spark Open always reasons before it answers.
            r = client.chat(provider, msg, purpose="smoke", max_tokens=2048)
            results.append({"provider": provider, "model": r.record.model, "reply": r.text.strip()[:40],
                            "tokens": (r.record.prompt_tokens, r.record.completion_tokens),
                            "latency_ms": r.record.latency_ms})
        except Exception as e:  # report, don't hide
            results.append({"provider": provider, "error": str(e)[:300]})
    for row in results:
        print(row)
    return results


@app.function(image=image, secrets=[keys], timeout=1800)
def demo() -> dict:
    """Live end to end on synthetic data: curate -> plan -> confirm -> run in sandbox -> draft ->
    sign-off -> Reviewer Assistant briefing (Muse Spark Open).

    Uses a throwaway folder, not the project volume. Prints a summary only.
    """
    import tempfile
    import time

    from saksi.workflow import SIGNOFF_QUESTIONS

    timings: dict[str, float] = {}

    def timed(stage, fn, *a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            timings[stage] = round(time.perf_counter() - t0, 1)

    from saksi import demo_data
    from saksi.canonical import sha256_bytes
    from saksi.workflow import Project, ledger

    root = tempfile.mkdtemp()
    p = Project(root, "live-demo")
    p.create("Synthetic GDM demo", "primary", {"actor_id": "author-a", "name": "Demo Author A"})
    p.register_human("author-a", {"actor_id": "author-b", "name": "Demo Author B"}, "author")
    p.register_human("author-a", {"actor_id": "reviewer-1", "name": "Demo Reviewer"}, "reviewer")
    for kind, name, data in (("dataset", "gdm_synthetic.csv", demo_data.make().to_csv(index=False).encode()),
                             ("protocol", "protocol.txt", demo_data.PROTOCOL.encode())):
        sha = sha256_bytes(data)
        (p.root / "materials").mkdir(parents=True, exist_ok=True)
        (p.root / "materials" / sha).write_bytes(data)
        p.upload_material("author-a", kind, name, sha, len(data), storage_ref=f"materials/{sha}")

    agents = _agents_factory(p)
    s = timed("curate", agents.curate, "author-a")
    rep = p.blobs.get_json(s.curator_reports[-1]["report_sha256"])
    print("\n== Data Curator ==\n", rep["summary"])
    for f in rep["flags"][:6]:
        print(f"  [{f.get('severity')}] {f.get('column')}: {f.get('issue')}")

    s = timed("plan", agents.propose_plan, "author-b")
    print("\n== Proposed plan ==")
    for aid in s.plan["order"]:
        a = s.plan["analyses"][aid]
        print(f"  {aid}: {a.get('title')} | quote verified: {a['protocol_quote_verified']}"
              f" | columns not in data: {a['columns_not_in_data']}")
    for author in ("author-a", "author-b"):
        s = p.confirm_plan(author, s.plan["sha"])

    s = timed("analyses", agents.run_planned, "author-a")
    print("\n== Analyses ==")
    for aid, a in s.analyses.items():
        ex = a["execution"]
        prov = ex.get("provenance", {})
        print(f"  {aid} [{a['label']}] {a['status']} | attempts: {[x['outcome'] for x in ex['attempts']]}"
              f" | backend: {ex['backend']} | network: {prov.get('network')} | sandbox: {prov.get('sandbox_object_id')}")
        print(f"     data hash matches upload: {prov.get('data_sha256') == ex['dataset_sha256']}"
              f" | packages: {prov.get('packages')}")
        if a["status"] == "ok":
            out = p.blobs.get_json(a["outputs_sha256"])
            print(f"     n analysed: {out.get('n_analysed')} | excluded: {out.get('n_excluded')}"
                  f" ({out.get('exclusion_reason')})")
            print(f"     model: {(out.get('model') or {}).get('formula_or_description')}")
            for e in out.get("estimates", [])[:6]:
                print(f"     {e.get('name')}: {e.get('estimate')} ({e.get('ci_lower')}, {e.get('ci_upper')})"
                      f" {e.get('measure')}")
            for note in (out.get("notes") or [])[:4]:
                print(f"     note: {str(note)[:200]}")
        else:
            print(f"     error: {a['error'][:300]}")

    planned_ok = all(any(a["plan_analysis_id"] == pid and a["status"] == "ok" for a in s.analyses.values())
                     for pid in s.plan_locked["order"])
    if planned_ok:
        for author in ("author-a", "author-b"):
            s = p.accept_analyses(author, s.analysis_set_sha())
        s = timed("draft", agents.write_draft, "author-a")
        d = s.drafts[s.draft_version]
        text = p.blobs.get_text(d["text_sha256"])
        print("\n== Draft v1 (Writing Agent) ==")
        print("\n".join("  " + ln for ln in text.splitlines()[:60]))
        c = d["checks"]
        print("\n== Writing attempts ==")
        for a in d.get("writer_attempts", []):
            f = p.blobs.get_json(a["findings_sha256"])
            print(f"  attempt {a['attempt']} ({a['prompt']['name']}): passed check = {a['passed_check']}"
                  f" | untraced {len(f.get('untraced', []))} | not reported {f.get('analyses_not_reported', [])}")
        print(f"\n== Number check (final draft) ==\n  numbers checked: {c['numbers_checked']} | untraced: {c['untraced']}"
              f" | exploratory not labelled: {c['exploratory_not_labelled']}"
              f" | analyses not reported: {c.get('analyses_not_reported')}"
              f" | unknown refs: {c['unknown_analysis_refs']} | all clear: {c['ok']}")
        for u in p.blobs.get_json(d["checks_sha256"])["untraced"][:8]:
            print(f"  UNTRACED {u['number']}: {u['sentence'][:160]}")

        # Demo authors sign off so the draft reaches review. (In real use each author reads it first.)
        answers = {q: "Demo run: checked against the computed outputs." for q in SIGNOFF_QUESTIONS}
        for author in ("author-a", "author-b"):
            p.open_draft(author, s.draft_version)
            s = p.sign_off(author, s.draft_version,
                           "Demo run: booking BMI and gestational diabetes, as reported in the Results.", answers)
        print(f"\n== Stage: {s.stage}, round {s.round} ==")
        try:
            s = timed("review_brief", agents.brief_reviewer, "reviewer-1")
            b = s.briefings[-1]
            full = p.blobs.get_json(b["briefing_sha256"])
            call = next(e.payload for e in p.log.events() if e.seq == b["llm_call_seq"])
            print(f"\n== Reviewer Assistant ({call['model']} via {call['provider']}) ==")
            print("  interval facts given (computed by code):")
            for f in full.get("interval_facts", []):
                print(f"     {f['analysis']}/{f['name']}: {f['ci_lower']:.4g} to {f['ci_upper']:.4g}"
                      f" | null {f['null_value']} | includes null: {f['interval_includes_null']}")
            print("  primary claim:", full["primary_claim"])
            print("  summary:", full["summary"])
            for pt in full["points_of_inquiry"]:
                print(f"  {pt['id']} [{pt['severity']}, {pt['category']}] quote verified: {pt['quote_verified']}")
                print(f"     \"{pt['quote'][:140]}\"")
                print(f"     Q: {pt['question'][:300]}")
        except Exception as e:  # report, don't hide
            print(f"\n(Reviewer Assistant failed: {str(e)[:300]})")
    else:
        print("\n(Draft skipped: a planned analysis did not succeed.)")

    calls = [e.payload for e in p.log.events() if e.type == "llm.called"]
    tokens = sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0) for c in calls)
    print("\n== Cost of the run ==")
    print("  seconds per stage:", timings)
    print("  tokens per agent:", {k: v["tokens"] for k, v in ledger(p.log.events()).items() if v["tokens"]})
    v = p.log.verify()
    summary = {"events": v.events, "chain_ok": v.ok, "model_calls": len(calls), "tokens": tokens,
               "analyses_ok": sum(a["status"] == "ok" for a in s.analyses.values()),
               "analyses_total": len(s.analyses), "stage": s.stage}
    print("\n== Record ==\n", summary, "\n  ledger:", {k: v["counts"] for k, v in ledger(p.log.events()).items()})
    return summary
