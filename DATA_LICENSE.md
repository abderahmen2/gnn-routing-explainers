# Vendored topology data

`data/topologies/` contains 229 network topologies redistributed here so the
dataset can be regenerated without depending on an external mirror staying
reachable. **The MIT licence in `LICENSE` covers the code in this repository,
not these files.**

Each file is NetworkX node-link JSON with geographic node positions and, per
edge, `dist`, the great-circle link length in km. `provenance.json` records
the source family and name of every file.

## Internet Topology Zoo — 203 files, `data/topologies/topozoo/`

Licensed **CC BY-SA**. Redistribution is permitted with attribution; the
share-alike term applies to the data and derivative data, not to this
repository's code. Cite:

> S. Knight, H. X. Nguyen, N. Falkner, R. Bowden and M. Roughan,
> "The Internet Topology Zoo", *IEEE Journal on Selected Areas in
> Communications*, 29(9):1765–1775, 2011.

## SNDlib — 26 files, `data/topologies/sndlib/`

SNDlib is free for research use and requires citation. Cite:

> S. Orlowski, R. Wessäly, M. Pióro and A. Tomaszewski, "SNDlib 1.0 —
> Survivable Network Design Library", *Networks*, 55(3):276–286, 2010.

If you intend to redistribute these files further, confirm SNDlib's current
terms at <http://sndlib.zib.de> first — they are less explicit about
redistribution than Topology Zoo's CC BY-SA.

## Mirror

Both families were obtained through
[`piotrjurkiewicz/topohub`](https://github.com/piotrjurkiewicz/topohub) (MIT),
a cleaned mirror, because topology-zoo.org was not reachable. **Cite the
underlying datasets above, not the mirror.**

## Removing the vendored data

If you would rather not redistribute it, delete the directory and let the
fetch script retrieve it:

```bash
git rm -r --cached data/topologies && rm -rf data/topologies
echo "data/topologies/" >> .gitignore
python scripts/fetch_topologies.py   # restores it locally
```
