"""Curated v2 dataset: enzymatic HIV-1 protease IC50 (ChEMBL CHEMBL243), scaffold split, train-only features.

Reuses the team's raw ChEMBL pull (Dataset/csv_files/hiv1_protease_01_bioactivity_raw.csv) and PaDEL PubChem
fingerprints (Dataset/csv_files/descriptors_output.csv). Original files are read, never modified.

Run:  python -m qlink_screen.data
"""
import json

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from . import DATA_V2, ROOT

RDLogger.DisableLog("rdApp.*")

RAW = ROOT / "Dataset" / "csv_files" / "hiv1_protease_01_bioactivity_raw.csv"
PADEL = ROOT / "Dataset" / "csv_files" / "descriptors_output.csv"

SPLIT_SEED = 2026
N_COMPACT = 16          # train-only PCA components fed to every compact-feature model
POTENT_PIC50 = 8.0      # primary screening endpoint: IC50 <= 10 nM (fixed before any v2 training/test eval)
CELL_WORDS = r"antiviral|cell|infected|in vivo|H9|MT-?4|CEM|PBMC"


def three_class(ic50_nm):
    """Team's original 3-class thresholds, unchanged: inactive >= 10 uM, active <= 1 uM."""
    return np.where(ic50_nm >= 10000, 0, np.where(ic50_nm <= 1000, 2, 1))


def standardize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol)


def curate():
    raw = pd.read_csv(RAW, low_memory=False)
    log = {"source": "ChEMBL web resource client, target CHEMBL243 (HIV-1 protease), standard_type=IC50",
           "raw_file": str(RAW.relative_to(ROOT)), "raw_rows": int(len(raw))}
    d = raw[(raw.standard_type == "IC50") & (raw.standard_units == "nM") & (raw.standard_relation == "=")
            & raw.standard_value.notna() & raw.canonical_smiles.notna() & (raw.standard_value > 0)]
    log["after_nM_equals_relation"] = int(len(d))
    d = d[d.data_validity_comment.isna()]
    log["after_drop_validity_flagged"] = int(len(d))
    d = d[(d.assay_type == "B") & d.bao_label.isin(["single protein format", "assay format"])]
    d = d[~d.assay_description.str.contains(CELL_WORDS, case=False, regex=True, na=False)]
    log["after_enzymatic_only"] = int(len(d))

    std = {s: standardize(s) for s in d.canonical_smiles.unique()}
    d = d.assign(std_smiles=d.canonical_smiles.map(lambda s: std[s][0]),
                 inchikey=d.canonical_smiles.map(lambda s: std[s][1]))
    d = d[d.std_smiles.notna()]
    log["after_rdkit_parse"] = int(len(d))
    d = d.assign(pIC50=9.0 - np.log10(d.standard_value.astype(float)))

    g = d.groupby("inchikey")
    mol = pd.DataFrame({
        "molecule_chembl_id": g.molecule_chembl_id.first(),
        "all_chembl_ids": g.molecule_chembl_id.agg(lambda s: ";".join(sorted(set(s)))),
        "smiles": g.std_smiles.first(),
        "pIC50": g.pIC50.median(),
        "pIC50_spread": g.pIC50.agg(lambda s: s.max() - s.min()),
        "n_measurements": g.size(),
        "n_assays": g.assay_chembl_id.nunique(),
    }).reset_index()
    log["unique_compounds_median_aggregated"] = int(len(mol))
    log["compounds_with_spread_gt_1_log"] = int((mol.pIC50_spread > 1).sum())

    padel = pd.read_csv(PADEL)
    have = set(padel.Name)
    mol["padel_id"] = mol.all_chembl_ids.map(lambda s: next((i for i in s.split(";") if i in have), None))
    log["dropped_no_padel_fingerprint"] = int(mol.padel_id.isna().sum())
    mol = mol[mol.padel_id.notna()].reset_index(drop=True)

    ic50 = 10 ** (9 - mol.pIC50)
    mol["y3"] = three_class(ic50)
    mol["potent"] = (mol.pIC50 >= POTENT_PIC50).astype(int)
    log["final_compounds"] = int(len(mol))
    log["class3_counts"] = {k: int(v) for k, v in zip(["inactive", "intermediate", "active"],
                                                      np.bincount(mol.y3, minlength=3))}
    log["potent_prevalence"] = round(float(mol.potent.mean()), 4)
    return mol, padel, log


def scaffold_split(mol, frac=(0.70, 0.15, 0.15), seed=SPLIT_SEED):
    """Randomized Bemis-Murcko scaffold split; scaffolds larger than half a test fold go to train."""
    scaf = mol.smiles.map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False))
    groups = scaf.groupby(scaf).groups
    rng = np.random.default_rng(seed)
    n = len(mol)
    big = [k for k, v in groups.items() if len(v) > frac[2] * n / 2]
    small = [k for k in groups if k not in big]
    rng.shuffle(small)
    split = pd.Series("train", index=mol.index)
    n_val = n_test = 0
    for k in small:
        idx = groups[k]
        if n_test + len(idx) <= frac[2] * n:
            split[idx] = "test"; n_test += len(idx)
        elif n_val + len(idx) <= frac[1] * n:
            split[idx] = "val"; n_val += len(idx)
    return split, scaf


def morgan(smiles, n_bits=2048):
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=n_bits)
    for i, s in enumerate(smiles):
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(s)), out[i])
    return out


def build():
    DATA_V2.mkdir(parents=True, exist_ok=True)
    mol, padel, log = curate()
    split, scaf = scaffold_split(mol)
    mol["scaffold"] = scaf
    mol["split"] = split
    for s in ["train", "val", "test"]:
        m = mol[mol.split == s]
        log[f"{s}_n"] = int(len(m))
        log[f"{s}_potent"] = int(m.potent.sum())
        log[f"{s}_class3"] = np.bincount(m.y3, minlength=3).tolist()
    assert not (set(mol[mol.split == "train"].scaffold) & set(mol[mol.split == "test"].scaffold))

    pub = padel.set_index("Name").loc[mol.padel_id].values.astype(np.float32)
    ecfp = morgan(mol.smiles.tolist())
    tr = (mol.split == "train").values

    # Train-only compact embedding shared by every compact-feature model (classical and quantum).
    vt = VarianceThreshold(0.01).fit(pub[tr])
    pca = PCA(N_COMPACT, random_state=0).fit(vt.transform(pub[tr]))
    z = pca.transform(vt.transform(pub))
    sc = StandardScaler().fit(z[tr])
    compact = sc.transform(z).astype(np.float32)
    log["compact"] = {"method": "VarianceThreshold(0.01) -> PCA(16) -> StandardScaler, all fit on train only",
                      "pubchem_bits_kept": int(vt.get_support().sum()),
                      "pca_explained_variance_ratio_sum": round(float(pca.explained_variance_ratio_.sum()), 4)}

    mol.drop(columns=["padel_id"]).to_csv(DATA_V2 / "curated.csv", index=False)
    mol[["molecule_chembl_id", "inchikey", "split"]].to_csv(DATA_V2 / "splits.csv", index=False)
    np.savez_compressed(DATA_V2 / "features.npz", pubchem=pub.astype(np.uint8), ecfp4=ecfp, compact=compact,
                        y3=mol.y3.values, potent=mol.potent.values, pIC50=mol.pIC50.values.astype(np.float32),
                        split=mol.split.values.astype(str))
    # Fitted transform, so unlabeled user molecules (PubChem FP) can be scored with the same embedding.
    np.savez_compressed(DATA_V2 / "compact_transform.npz", support=vt.get_support(), pca_mean=pca.mean_,
                        pca_components=pca.components_, sc_mean=sc.mean_, sc_scale=sc.scale_)
    log["split"] = {"method": "randomized Bemis-Murcko scaffold split 70/15/15", "seed": SPLIT_SEED}
    log["endpoints"] = {"primary_screening": f"potent = median pIC50 >= {POTENT_PIC50} (IC50 <= 10 nM)",
                        "secondary_classification": "team 3-class: inactive IC50>=10uM, intermediate, active<=1uM"}
    (DATA_V2 / "provenance.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    build()
