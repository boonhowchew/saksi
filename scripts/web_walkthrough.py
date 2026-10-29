# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew
"""Drive the whole flow in a headless browser against scripts/offline_server.py:
create a project, add a co-author and a reviewer, upload, curate, plan, confirm,
analyse, accept, draft, sign off, two review rounds, attest. Saves screenshots.

  python scripts/offline_server.py &        # fresh server for each run
  python scripts/web_walkthrough.py OUT_DIR
"""
import os, re, sys, random, tempfile
from pathlib import Path
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saksi import demo_data
DATA = OUT / "gdm_synthetic.csv"; demo_data.make().to_csv(DATA, index=False)
PROTO = OUT / "protocol.txt"; PROTO.write_text(demo_data.PROTOCOL)
LIMIT = "The number of GDM events was small relative to the number of model parameters, so the estimates may be unstable."
PID = f"gdm-demo-{random.randint(1000,9999)}"
from playwright.sync_api import sync_playwright
U = "http://127.0.0.1:8765/"
errors = []
def shot(page, name): page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
def note(page):  # last status message
    el = page.locator("#status .msg")
    return el.inner_text() if el.count() else ""
def click(page, text, exact=False):
    page.get_by_role("button", name=text, exact=exact).first.click()
    page.wait_for_function("() => !document.querySelector('.btn.busy')", timeout=60000)
    page.wait_for_timeout(250)
def signin(page, pid, token):
    page.goto(U)
    page.fill("input[name=pid]", pid); page.fill("input[name=token]", token)
    page.locator("form[data-form=signin] button").click(); page.wait_for_selector(".stages")
def signout(page):
    page.get_by_role("button", name="Sign out").click(); page.wait_for_selector(".gate")

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1360, "height": 900})
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(U); page.wait_for_selector(".gate"); shot(page, "01-gate")
    page.get_by_text("Start a new project").click()
    f = page.locator("form[data-form=create]")
    f.locator("input[name=pid]").fill(PID); f.locator("input[name=title]").fill("Booking BMI and gestational diabetes")
    f.locator("input[name=name]").fill("Boon-How Chew"); f.locator("input[name=actor_id]").fill("bhc")
    f.locator("input[name=orcid]").fill("0000-0002-8627-6248"); f.locator("input[name=affiliation]").fill("UPM")
    f.get_by_role("button", name="Start project").click(); page.wait_for_selector(".stages")
    tok_a = page.locator(".secret").inner_text().strip()
    shot(page, "02-new-project")
    # add co-author and reviewer
    toks = {}
    for nm, aid, role in (("Siti Aminah", "siti", "author"), ("Dr Reviewer", "rev1", "reviewer")):
        page.get_by_role("tab", name="People").click()
        page.fill("form[data-form=addperson] input[name=name]", nm); page.fill("form[data-form=addperson] input[name=actor_id]", aid)
        page.select_option("form[data-form=addperson] select[name=role]", role)
        click(page, "Add person"); toks[aid] = page.locator(".secret").inner_text().strip()
    print("people ok", note(page))
    # upload
    page.get_by_role("tab", name="Data").click()
    for kind, path in (("dataset", str(DATA)), ("protocol", str(PROTO))):
        page.select_option("form[data-form=upload] select[name=kind]", kind)
        page.set_input_files("form[data-form=upload] input[type=file]", path)
        click(page, "Upload")
    click(page, "Ask the Data Curator to check the dataset"); shot(page, "03-curator")
    print("curator:", note(page))
    page.get_by_role("tab", name="Plan").click()
    click(page, "Ask the Analysis Agent to draft the plan"); click(page, "Confirm this plan"); shot(page, "04-plan")
    print("plan:", note(page))
    # co-author confirms
    signout(page); signin(page, PID, toks["siti"])
    page.get_by_role("tab", name="Plan").click(); click(page, "Confirm this plan")
    print("stage after both confirm:", page.locator(".stages li.now").inner_text())
    page.get_by_role("tab", name="Analyses").click(); click(page, "Run the planned analyses"); click(page, "Accept these results")
    signout(page); page.locator("[data-action=resume]").first.click(); page.wait_for_selector(".stages")  # resume saved identity (author A)
    print("acting as:", page.locator(".next strong").inner_text())
    page.get_by_role("tab", name="Analyses").click(); page.get_by_text("How it ran").first.click(); shot(page, "05-analyses")
    click(page, "Accept these results")
    page.get_by_role("tab", name="Draft").click(); click(page, "Ask the Writing Agent for the first draft"); shot(page, "06-draft")
    print("draft:", page.locator(".msg.ok, .msg.warn").first.inner_text()[:120])
    # sign off: A then B (switch identity via selector)
    def signoff(pg):
        pg.get_by_role("tab", name="Draft").click(); pg.wait_for_timeout(400)
        pg.fill("form[data-form=signoff] textarea[name=claim]", "Booking BMI was not clearly associated with gestational diabetes after adjustment in this cohort.")
        for i in range(3): pg.fill(f"form[data-form=signoff] textarea[name=q{i}]", "Checked against the results table and the number check.")
        click(pg, "Sign off version")
    signoff(page)
    page.select_option("select[data-change=switch]", "siti"); page.wait_for_timeout(600)
    print("switched to:", page.locator(".next strong").inner_text())
    signoff(page)
    print("stage:", page.locator(".stages li.now").inner_text())
    # reviewer
    signout(page); signin(page, PID, toks["rev1"])
    print("reviewer next:", page.locator(".next p").inner_text())
    page.get_by_role("tab", name="Review").click(); click(page, "Ask for a briefing on version 1"); shot(page, "07-briefing")
    page.get_by_role("button", name="Use as a comment").first.click(); page.wait_for_timeout(400)
    click(page, "Add comment"); click(page, "Ask for revision")
    print("after revision:", page.locator(".stages li.now").inner_text())
    # authors respond
    signout(page); page.locator("[data-action=resume]").first.click(); page.wait_for_selector(".stages")
    page.get_by_role("tab", name="Draft").click(); page.wait_for_timeout(400)
    ta = page.locator("form[data-form=edit] textarea[name=text]")
    ta.fill(ta.input_value() + "\n" + LIMIT + "\n")
    page.fill("form[data-form=edit] input[name=summary]", "Added the small-events limitation (R1)")
    click(page, "Save as version 2")
    page.get_by_role("tab", name="Review").click()
    page.fill("form[data-form=respond] input[name=note]", "Added a limitation to the Discussion.")
    click(page, "Respond"); signoff(page)
    page.select_option("select[data-change=switch]", "siti"); page.wait_for_timeout(600); signoff(page)
    page.select_option("select[data-change=switch]", "bhc"); page.wait_for_timeout(600)
    page.get_by_role("tab", name="Review").click(); click(page, "Send the response to the reviewer")
    print("after response:", page.locator(".stages li.now").inner_text())
    signout(page); signin(page, PID, toks["rev1"])
    page.get_by_role("tab", name="Review").click(); click(page, "Ask for a briefing on version 2"); shot(page, "08-round2")
    click(page, "Give verdict"); click(page, "Accept the manuscript")
    signout(page); page.locator("[data-action=resume]").first.click(); page.wait_for_selector(".stages")
    for who in ("bhc", "siti"):
        page.select_option("select[data-change=switch]", who); page.wait_for_timeout(600)
        page.get_by_role("tab", name="Review").click()
        page.fill("form[data-form=attest] textarea[name=statement]", "I have read the final version and stand behind it.")
        click(page, "Attest version 2")
    print("final stage:", page.locator(".stages li.now").inner_text(), "|", page.locator(".seal").inner_text()[:60])
    page.get_by_role("tab", name="Record").click(); page.wait_for_timeout(500); shot(page, "09-record")
    page.get_by_role("tab", name="Draft").click(); page.wait_for_timeout(400); shot(page, "10-final-draft")
    m = page; m.set_viewport_size({"width": 390, "height": 844})
    m.get_by_role("tab", name="Review").click(); m.wait_for_timeout(400); m.screenshot(path=str(OUT / "11-mobile.png"), full_page=False)
    print("mobile horizontal overflow:", m.evaluate("document.documentElement.scrollWidth > window.innerWidth"))
    b.close()
print("console errors:", errors)
print("screenshots in", OUT)
sys.exit(1 if errors else 0)
