"""Drug mechanism-of-action reference used to catch invented mechanisms.

Source: Broad Institute Drug Repurposing Hub (CC BY 4.0), downloaded on first use and cached locally.
If the download fails, the mechanism check is switched off and the report says so.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import httpx

REFERENCE_FILE = "repurposing_drugs.txt"

# Words that say nothing about *which* mechanism ("HDAC inhibitor" -> keep "hdac").
GENERIC = {
    "inhibitor", "inhibitors", "agonist", "antagonist", "activator", "blocker", "modulator", "receptor",
    "protein", "kinase", "channel", "enzyme", "stimulant", "analog", "derivative", "ligand", "pathway",
    "signaling", "family", "release", "reuptake", "uptake", "partial", "inverse", "positive", "negative",
    "allosteric", "selective", "other", "factor", "dependent", "like", "type", "subunit", "and", "the",
    "agent", "agents", "drug", "compound", "tyrosine", "serine", "threonine",
}

# Common ways a model names the same mechanism differently from the reference wording.
SYNONYMS: dict[str, list[str]] = {
    "hdac": ["histone deacetylase"],
    "tubulin": ["microtubule", "mitotic spindle", "spindle poison"],
    "kif11": ["eg5", "ksp", "kinesin spindle", "kinesin-5", "kinesin"],
    "kinesin": ["eg5", "ksp", "kif11"],
    "nampt": ["nicotinamide phosphoribosyltransferase", "nad+ biosynthesis", "nad biosynthesis", "nad+ salvage"],
    "topoisomerase": ["top1", "top2", "topo i", "topo ii"],
    "proteasome": ["psmb5", "26s"],
    "hsp": ["heat shock protein"],
    "parp": ["poly(adp-ribose) polymerase", "poly adp-ribose"],
    "plk": ["polo-like kinase"],
    "cdk": ["cyclin-dependent kinase"],
    "egfr": ["epidermal growth factor receptor", "erbb1", "her1"],
    "erbb2": ["her2"],
    "mtor": ["mammalian target of rapamycin", "mtorc1", "mtorc2"],
    "pi3k": ["phosphoinositide 3-kinase", "phosphatidylinositol 3-kinase", "pik3ca", "pi3-kinase"],
    "pik3ca": ["pi3k"],
    "mek": ["map2k", "mapk/erk kinase"],
    "map2k": ["mek"],
    "raf": ["braf", "craf"],
    "braf": ["raf"],
    "bcl": ["bcl-2", "bcl2", "bcl-xl"],
    "aurora": ["aurka", "aurkb"],
    "aurk": ["aurora"],
    "dna": ["dna synthesis", "dna damage", "alkylat", "intercalat", "crosslink"],
    "dihydrofolate": ["dhfr", "antifolate"],
    "thymidylate": ["tyms", "antimetabolite"],
    "ribonucleotide": ["rrm1", "rrm2", "nucleoside analog", "nucleoside analogue", "antimetabolite", "chain termination"],
    "jak": ["janus kinase"],
    "abl": ["bcr-abl", "bcr/abl"],
    "fgfr": ["fibroblast growth factor receptor"],
    "vegfr": ["vascular endothelial growth factor receptor", "kdr"],
    "igf": ["igf1r", "igf-1r", "insulin-like growth factor"],
    "atpase": ["na+/k+", "sodium-potassium", "cardiac glycoside"],
    "glucocorticoid": ["nr3c1", "corticosteroid"],
    "nedd8": ["neddylation"],
    "bromodomain": ["brd4", "bet"],
    "chk": ["chek1", "checkpoint kinase"],
    "smoothened": ["smo", "hedgehog"],
    "ikk": ["ikbkb", "ikk-beta", "ikk2", "nf-kb", "nf-κb"],
}

SALT_WORDS = re.compile(r"\b(free base|hydrochloride|dihydrochloride|hcl|mesylate|maleate|sodium|tosylate|citrate)\b", re.I)


def norm_name(name: str) -> str:
    name = re.sub(r"\(.*?\)", "", name)
    name = SALT_WORDS.sub("", name)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def drug_components(name: str) -> list[str]:
    """'JQ-1:carboplatin (1:1 mol/mol)' -> ['JQ-1', 'carboplatin']. About 10% of the rows are combinations."""
    name = re.sub(r"\(.*?\)", "", name)
    return [c.strip() for c in re.split(r"[:+]", name) if c.strip()]


@dataclass(frozen=True)
class MoaEntry:
    drug: str
    moa: tuple[str, ...]
    targets: tuple[str, ...]

    def keywords(self) -> set[str]:
        kws: set[str] = set()
        for phrase in self.moa:
            words = [w for w in re.split(r"[\s/,]+", phrase.lower()) if w]
            specific = [w for w in words if w not in GENERIC and len(w) >= 3]
            kws.update(specific)
            if len(specific) > 1:
                kws.add(" ".join(specific))
        for target in self.targets:
            t = target.lower()
            kws.add(t)
            family = re.sub(r"\d+[a-z]?$", "", t)
            if len(family) >= 3:
                kws.add(family)
        for kw in list(kws):
            kws.update(SYNONYMS.get(kw, []))
        return kws


class MoaReference:
    def __init__(self, entries: dict[str, MoaEntry]):
        self.entries = entries

    @classmethod
    def from_file(cls, path: str | Path) -> "MoaReference":
        entries: dict[str, MoaEntry] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("!") or line.startswith("pert_iname"):
                continue
            parts = line.split("\t")
            if len(parts) < 3 or not parts[2].strip():
                continue
            moa = tuple(m.strip() for m in parts[2].split("|") if m.strip())
            targets = tuple(t.strip() for t in parts[3].split("|") if t.strip()) if len(parts) > 3 else ()
            entries[norm_name(parts[0])] = MoaEntry(parts[0], moa, targets)
        return cls(entries)

    def lookup(self, drug: str | None) -> MoaEntry | None:
        return self.entries.get(norm_name(drug)) if drug else None

    @staticmethod
    def mentions(text: str, keywords: set[str]) -> list[str]:
        low = text.lower()
        hits = []
        for kw in keywords:
            if len(kw) <= 4:  # short symbols need word boundaries ("ret", "kit"); allow plurals ("HDACs")
                if re.search(rf"(?<![a-z0-9]){re.escape(kw)}s?(?![a-z0-9])", low):
                    hits.append(kw)
            elif kw in low:
                hits.append(kw)
        return sorted(hits)


def load_reference(reference_dir: str | Path, url: str) -> MoaReference | None:
    path = Path(reference_dir) / REFERENCE_FILE
    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            resp = httpx.get(url, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            path.write_text(resp.text, encoding="utf-8")
        except (httpx.HTTPError, OSError) as e:
            warnings.warn(f"Mechanism reference unavailable ({e!r}); mechanism check disabled.", stacklevel=2)
            return None
    return MoaReference.from_file(path)
