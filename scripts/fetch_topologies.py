#!/usr/bin/env python3
"""
Download real network topologies and cache them locally.

Source
------
`piotrjurkiewicz/topohub` (MIT), a curated, cleaned mirror of the Internet
Topology Zoo, SNDlib, CAIDA and others. The original topology-zoo.org is not
reachable from this environment; the mirror is used instead and the provenance
of every file is recorded in `provenance.json` so the paper can cite the
underlying datasets rather than the mirror.

Cite the underlying sources, not topohub:
  Internet Topology Zoo -- Knight et al., "The Internet Topology Zoo",
    IEEE JSAC 29(9), 2011.
  SNDlib -- Orlowski et al., "SNDlib 1.0 -- Survivable Network Design Library",
    Networks 55(3), 2010.

What the files give us
----------------------
Each topology is NetworkX node-link JSON with geographic node positions and,
for every edge, `dist`: the great-circle link length in km. That is what makes
these usable here -- the link delay is a *physical* quantity (length divided by
the speed of light in fibre) rather than the U[0.8, 1.2] draw the synthetic
generator uses. SNDlib entries additionally carry real traffic demand matrices.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = "piotrjurkiewicz/topohub"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/data"
API = f"https://api.github.com/repos/{REPO}/contents/data"

SOURCE_CITATION = {
    "topozoo": "Knight et al., The Internet Topology Zoo, IEEE JSAC 29(9), 2011",
    "sndlib": "Orlowski et al., SNDlib 1.0, Networks 55(3), 2010",
}


def _get_json(url: str, timeout: int = 30):
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return json.load(fh)


def list_topologies(family: str) -> list:
    entries = _get_json(f"{API}/{family}")
    return sorted(e["name"][:-5] for e in entries if e["name"].endswith(".json"))


def fetch_one(family: str, name: str, outdir: str) -> tuple:
    dest = os.path.join(outdir, family, f"{name}.json")
    if os.path.exists(dest):
        return name, "cached", os.path.getsize(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        blob = _get_json(f"{RAW}/{family}/{name}.json")
    except Exception as exc:  # noqa: BLE001
        return name, f"FAILED {type(exc).__name__}", 0
    with open(dest, "w") as fh:
        json.dump(blob, fh)
    return name, "ok", os.path.getsize(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/topologies")
    ap.add_argument("--families", nargs="+", default=["topozoo", "sndlib"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest = {}
    for fam in args.families:
        names = list_topologies(fam)
        print(f"{fam}: {len(names)} topologies")
        ok = failed = 0
        with cf.ThreadPoolExecutor(args.workers) as ex:
            for name, status, _size in ex.map(
                lambda n, f=fam: fetch_one(f, n, args.out), names
            ):
                if status.startswith("FAILED"):
                    failed += 1
                    print(f"  {name}: {status}")
                else:
                    ok += 1
        print(f"  {ok} fetched/cached, {failed} failed")
        manifest[fam] = {
            "count": ok,
            "names": names,
            "cite": SOURCE_CITATION.get(fam, ""),
        }

    prov = {
        "mirror_repo": REPO,
        "mirror_license": "MIT",
        "mirror_url": f"https://github.com/{REPO}",
        "note": ("topohub is a mirror. Cite the underlying datasets below, not "
                 "the mirror. Topology Zoo is CC BY-SA; check the licence of "
                 "each source before redistributing these files."),
        "families": manifest,
    }
    with open(os.path.join(args.out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"\nwrote {args.out}/provenance.json")


if __name__ == "__main__":
    main()
