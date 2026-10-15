"""Morning scan, meant to be run by Windows Task Scheduler (see install_daily_task.ps1).

    .venv\\Scripts\\python daily_scan.py                 # scan, Claude commentary, notification
    .venv\\Scripts\\python daily_scan.py --no-commentary --no-notify
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ledger.daily import run_scan  # noqa: E402
from ledger.db import DEFAULT_DB, Store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Stock Ledger daily trend scan")
    ap.add_argument("--no-commentary", action="store_true", help="skip Claude's written commentary")
    ap.add_argument("--no-notify", action="store_true", help="skip the Windows notification")
    args = ap.parse_args()

    log_file = Path(DEFAULT_DB).parent / "scan.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
    store = Store()
    try:
        run_id = run_scan(store, with_commentary=False if args.no_commentary else None, notify=not args.no_notify)
    except Exception as exc:
        logging.exception("daily scan failed")
        if not args.no_notify:
            from ledger.daily import toast
            toast("Stock Ledger: morning scan failed",
                  f"{type(exc).__name__}. Details in data/scan.log. You can run it from the Daily ideas page.")
        return 1
    run = store.scan_runs(1).iloc[0]
    logging.info("run %s: %s, %s/%s in uptrend, %.0fs. %s %s", run_id, run["status"], run["n_eligible"],
                 run["n_scanned"], run["duration_s"] or 0, run["notes"] or "", run["commentary_note"] or "")
    return 0 if run["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
