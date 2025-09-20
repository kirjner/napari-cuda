"""Drive multiscale policy intents against a running napari-cuda server."""

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
    policies: List[str],
    levels: List[int],
    zoom_factors: List[float],
    output_prefix: Path,
    idle_delay: float = 0.25,
    level_delay: float = 0.0,
    prime_levels: bool = False,
    verbose: bool = False,
) -> None:
    policies = [p for p in (str(pol).lower() for pol in policies) if p]
    policy = policies[0] if policies else "oversampling"

    records: List[Dict[str, object]] = []
    scenes: List[dict] = []

    def on_scene(spec) -> None:
        try:
            scenes.append(spec.to_dict())
        except Exception:
            scenes.append({"error": "scene serialization failed"})

    controller = StateController(host, state_port, on_scene_spec=on_scene)
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

    send({"type": "multiscale.intent.set_policy", "policy": policy}, f"set_policy({policy})")

    if prime_levels and levels:
        seen: set[int] = set()
        prime_sequence: list[int] = []
        for raw in levels:
            lvl = int(raw)
            if lvl in seen:
                continue
            seen.add(lvl)
            prime_sequence.append(lvl)
        for level in prime_sequence:
            send({"type": "multiscale.intent.set_level", "level": int(level)}, f"prime:set_level({level})")
            if level_delay > 0:
                time.sleep(level_delay)

    anchors = [(960.0, 540.0), (540.0, 300.0)]
    for idx, factor in enumerate(zoom_factors):
        anchor = anchors[idx % len(anchors)]
        send(
            {
                "type": "camera.zoom_at",
                "factor": float(factor),
                "anchor_px": [float(anchor[0]), float(anchor[1])],
            },
            f"camera.zoom_at({factor})",
        )

    summary = {
        "commands": records,
        "scenes": scenes,
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
    parser.add_argument("--policies", nargs="*", default=["oversampling"], help="Policy name (oversampling; 'latency' kept as alias)")
    parser.add_argument("--levels", nargs="*", type=int, default=[2, 1, 0], help="Level sequence used for optional priming")
    parser.add_argument("--zoom-factors", nargs="*", type=float, default=[0.6, 1.0, 1.8, 3.0])
    parser.add_argument("--output-prefix", type=Path, default=Path("tmp/policy_harness"))
    parser.add_argument("--idle-delay", type=float, default=0.25, help="Seconds to sleep after each command")
    parser.add_argument("--level-delay", type=float, default=0.0, help="Extra pause between prime level commands")
    parser.add_argument("--prime-levels", action="store_true", help="Cycle through --levels before sending zoom commands")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    run_harness(
        host=str(args.host),
        state_port=int(args.state_port),
        metrics_url=str(args.metrics_url),
        policies=list(args.policies),
        levels=list(args.levels),
        zoom_factors=list(args.zoom_factors),
        output_prefix=Path(args.output_prefix),
        idle_delay=float(args.idle_delay),
        level_delay=float(args.level_delay),
        prime_levels=bool(args.prime_levels),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()
