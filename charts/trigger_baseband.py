#!/usr/bin/env python3
"""Trigger CHARTS baseband capture or inspect capture and disk-write progress."""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request


def request_json(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode(errors="replace")) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:12048/charts/baseband")
    parser.add_argument("--status", action="store_true", help="Show status without triggering")
    parser.add_argument("--event-id", type=int)
    parser.add_argument("--duration", type=float, help="Capture duration in seconds")
    parser.add_argument("--mode", choices=["next", "previous", "spectrum"], default="next")
    parser.add_argument("--start-spectrum", type=int)
    parser.add_argument("--wait", action="store_true", help="Wait for successful disk writes")
    parser.add_argument("--wait-timeout", type=float, default=180)
    args = parser.parse_args()
    if not math.isfinite(args.wait_timeout) or args.wait_timeout <= 0:
        parser.error("--wait-timeout must be positive and finite")
    if args.status:
        print(json.dumps(request_json(args.url), indent=2))
        return 0
    if args.event_id is None or not 0 <= args.event_id < 2**64:
        parser.error("--event-id must be a nonnegative 64-bit integer")
    if args.duration is None or not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be positive and finite")
    if args.mode == "spectrum" and (args.start_spectrum is None or args.start_spectrum < 0):
        parser.error("--mode spectrum requires a nonnegative --start-spectrum")
    if args.mode != "spectrum" and args.start_spectrum is not None:
        parser.error("--start-spectrum requires --mode spectrum")
    payload = {
        "event_id": args.event_id,
        "duration_seconds": args.duration,
        "mode": args.mode,
    }
    if args.start_spectrum is not None:
        payload["start_spectrum"] = args.start_spectrum
    print(json.dumps(request_json(args.url, payload), indent=2), flush=True)
    if not args.wait:
        return 0
    deadline = time.monotonic() + args.wait_timeout
    while time.monotonic() < deadline:
        state = request_json(args.url)
        for event in state["events"]:
            if event["event_id"] == args.event_id and event["status"] in ("done", "error"):
                print(json.dumps(event, indent=2))
                return 0 if event["status"] == "done" else 1
        time.sleep(0.5)
    raise RuntimeError("Timed out waiting for capture completion; query --status for its state")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, urllib.error.URLError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
