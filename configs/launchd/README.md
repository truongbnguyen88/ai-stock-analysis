# Quarterly RAG refresh (macOS launchd) — RAG_TODO 9e

Schedule the SEC-filings refresh so the grounded-research corpus stays current without manual
runs. The job runs **locally** (not in CI) on purpose: the production vector store lives in this
repo's gitignored `data/` tree and the production embedder (voyage-4) reads `VOYAGE_API_KEY` from
`.env` — both are local, so a local scheduler is the right fit (the CI-train/local-serve model-retrain
pattern doesn't transfer). What it runs each quarter:

```bash
make refresh-filings        # = python -m stock_agent documents refresh --all --months 6
```

`documents refresh` pulls only newly-filed documents (manifest-idempotent) and **incrementally**
ingests them (only *new* chunks are embedded), so a quarterly run costs a small fraction of a full
rebuild and stays well inside the Voyage free pool.

## Install

```bash
# 1. Fill in the two placeholders (repo path + the venv python) into a real plist.
REPO="$(pwd)"
sed -e "s|__REPO__|$REPO|g" -e "s|__PYTHON__|$REPO/.venv/bin/python|g" \
    configs/launchd/com.stock-agent.refresh-filings.plist.template \
    > ~/Library/LaunchAgents/com.stock-agent.refresh-filings.plist

# 2. Load it (macOS).
launchctl load ~/Library/LaunchAgents/com.stock-agent.refresh-filings.plist

# 3. (optional) Trigger once now to verify it works end-to-end.
launchctl start com.stock-agent.refresh-filings
tail -f outputs/logs/refresh-filings.out.log
```

It fires at 09:00 on the 1st of Jan / Apr / Jul / Oct. A missed run (laptop asleep) is harmless —
the next run's 6-month look-back window catches anything skipped, and both download and ingest are
idempotent.

## Manage

```bash
launchctl list | grep stock-agent                                   # is it registered?
launchctl unload ~/Library/LaunchAgents/com.stock-agent.refresh-filings.plist   # disable
```

Prerequisites: `pip install -e ".[rag,voyage]"`, `SEC_USER_AGENT` set in `.env`, and the corpus
already bootstrapped (`documents download-sec --all` + `documents ingest --all`). Monthly cadence
also works — just change `StartCalendarInterval` to a single `Day = 1` entry.

> **Linux/cron alternative:** add `0 9 1 1,4,7,10 *  cd /path/to/repo && make refresh-filings` to
> your crontab (the command is platform-agnostic; only this launchd wrapper is macOS-specific).
