"""Simplest zoom harness: reset, set policy, zoom in xN, zoom out xM.

Radically simplified for predictable, readable runs:
- No priming, no forced level overrides
- One policy: oversampling
- Fixed center anchor
- Minimal outputs: sent commands + optional metrics snapshot
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

from napari_cuda.client.streaming.controllers import StateController

logger = logging.getLogger(__name__)


def _await_connection(channel, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if getattr(channel, "_out_q", None) is not None:
            return
        time.sleep(0.05)
    raise RuntimeError("state channel did not become ready")


def _fetch_metrics(url: str, timeout: float = 5.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # type: ignore[arg-type]
            return json.load(resp)
    except Exception as exc:  # pragma: no cover - network failure
        return {"error": f"metrics fetch failed: {exc}"}


def run_harness(
    host: str,
    state_port: int,
    metrics_url: str,
    output_prefix: Path,
    factor_in: float = 0.5,
    repeats_in: int = 4,
    factor_out: float = 2.0,
    repeats_out: int = 4,
    idle_delay: float = 0.4,
    reset_camera: bool = True,
    finalize_return: bool = True,
    final_zoom: float = 1.01,
    verbose: bool = False,
) -> None:
    records: List[Dict[str, object]] = []

    controller = StateController(host, state_port)
    channel, _thread = controller.start()
    _await_connection(channel)

    seq = 1

    def send(payload: Dict[str, object], label: str) -> None:
        nonlocal seq
        message = {
            "client_id": "policy-harness",
            "client_seq": seq,
            "origin": "harness",
            **payload,
        }
        ok = bool(channel.send_json(message))
        if verbose:
            logger.info("[%03d] %s -> %s", seq, label, ok)
        records.append({"seq": seq, "label": label, "ok": ok, "payload": message})
        seq += 1
        if idle_delay > 0:
            time.sleep(idle_delay)

    # Known baseline: reset camera and set oversampling policy
    if reset_camera:
        send({"type": "camera.reset"}, "camera.reset")
    send({"type": "multiscale.intent.set_policy", "policy": "oversampling"}, "set_policy(oversampling)")

    # Zoom in (factor < 1) N times, then zoom out (factor > 1) M times
    anchor = [960.0, 540.0]
    for _ in range(max(0, int(repeats_in))):
        send({"type": "camera.zoom_at", "factor": float(factor_in), "anchor_px": anchor}, f"zoom_in({factor_in})")
    for _ in range(max(0, int(repeats_out))):
        send({"type": "camera.zoom_at", "factor": float(factor_out), "anchor_px": anchor}, f"zoom_out({factor_out})")

    # Finale: return to a known baseline and trigger one policy eval at full view
    if finalize_return:
        send({"type": "camera.reset"}, "camera.reset")
        # Send two tiny zooms to allow stepwise clamp to traverse 0->1 and 1->2 if needed
        send({"type": "camera.zoom_at", "factor": float(final_zoom), "anchor_px": anchor}, f"final_zoom({final_zoom})")
        send({"type": "camera.zoom_at", "factor": float(final_zoom), "anchor_px": anchor}, f"final_zoom({final_zoom})")

    summary = {
        "commands": records,
        "metrics": _fetch_metrics(metrics_url),
    }

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_prefix.with_suffix(".json")
    output_path.write_text(json.dumps(summary, indent=2))
    if verbose:
        logger.info("summary saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=8081)
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8083/metrics.json")
    # Simplified zoom plan: in xN, out xM, then finalize
    parser.add_argument("--factor-in", type=float, default=0.5, help="Zoom-in factor (<1)")
    parser.add_argument("--repeats-in", type=int, default=4, help="Number of zoom-in steps")
    parser.add_argument("--factor-out", type=float, default=2.0, help="Zoom-out factor (>1)")
    parser.add_argument("--repeats-out", type=int, default=4, help="Number of zoom-out steps")
    parser.add_argument("--output-prefix", type=Path, default=Path("tmp/policy_harness"))
    parser.add_argument("--idle-delay", type=float, default=0.25, help="Seconds to sleep after each command")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reset-camera", action="store_true", help="Send camera.reset before zooms (default: off)")
    parser.add_argument("--finalize-return", action="store_true", help="After sweeps: camera.reset then a tiny zoom to trigger policy eval at full view")
    parser.add_argument("--final-zoom", type=float, default=1.01, help="Final tiny zoom factor (>0) used after reset to trigger selection")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    run_harness(
        host=str(args.host),
        state_port=int(args.state_port),
        metrics_url=str(args.metrics_url),
        output_prefix=Path(args.output_prefix),
        factor_in=float(args.factor_in),
        repeats_in=int(args.repeats_in),
        factor_out=float(args.factor_out),
        repeats_out=int(args.repeats_out),
        idle_delay=float(args.idle_delay),
        reset_camera=bool(args.reset_camera),
        finalize_return=bool(args.finalize_return),
        final_zoom=float(args.final_zoom),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()
