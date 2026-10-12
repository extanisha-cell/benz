"""Optional chemical-graph comparison with RDKit.

Nothing here reads images: it compares two SMILES strings that someone already extracted (for example with an
optical structure recognition tool) and vouches for. When RDKit is not installed, or a SMILES is missing, the
result is "unavailable" / "not_provided" and the decision layer ignores it, so the pipeline behaves the same
without RDKit. A "match" or "mismatch" is only as reliable as the extraction behind the SMILES.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphResult:
    status: str  # "match" | "mismatch" | "unavailable" | "unparseable" | "not_provided"
    detail: str = ""
    categories: tuple = ()  # chemical categories (policy.CHEMICAL_CATEGORIES) when status is "mismatch"

    def label(self) -> str:
        return f"{self.status}: {self.detail}" if self.detail else self.status


NOT_PROVIDED = GraphResult("not_provided")


def rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return False
    return True


def compare_smiles(reference: str | None, annotated: str | None) -> GraphResult:
    """Same molecule? Stereochemistry, isotopes and charges count; atom order and writing style do not."""
    if not (reference or "").strip() or not (annotated or "").strip():
        return NOT_PROVIDED
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        return GraphResult("unavailable", "rdkit is not installed")
    RDLogger.DisableLog("rdApp.*")
    mol_ref, mol_ann = Chem.MolFromSmiles(reference.strip()), Chem.MolFromSmiles(annotated.strip())
    if mol_ref is None or mol_ann is None:
        which = " and ".join(n for n, m in (("reference", mol_ref), ("annotated", mol_ann)) if m is None)
        return GraphResult("unparseable", f"RDKit could not parse the {which} SMILES")
    if Chem.MolToSmiles(mol_ref, isomericSmiles=True) == Chem.MolToSmiles(mol_ann, isomericSmiles=True):
        return GraphResult("match", "identical canonical isomeric SMILES")
    if Chem.MolToSmiles(mol_ref, isomericSmiles=False) == Chem.MolToSmiles(mol_ann, isomericSmiles=False):
        return GraphResult("mismatch", "same connectivity, different stereochemistry or isotopes", ("stereochemistry",))
    return GraphResult("mismatch", "different molecular graph", ("other_chemical",))


def load_smiles(path: str) -> dict[str, tuple[str, str]]:
    """IDs -> (reference SMILES, annotated SMILES) from a CSV with columns IDs, Samples_SMILES, Annotated_SMILES.
    The input CSV of the pipeline may carry these two columns; the pipeline ignores them unless asked."""
    out: dict[str, tuple[str, str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        names = {(n or "").strip().lower(): n for n in (reader.fieldnames or [])}
        for needed in ("ids", "samples_smiles", "annotated_smiles"):
            if needed not in names:
                raise ValueError(f"{path} has no {needed!r} column")
        for row in reader:
            out[(row[names["ids"]] or "").strip()] = ((row[names["samples_smiles"]] or "").strip(), (row[names["annotated_smiles"]] or "").strip())
    return out
