#!/usr/bin/env python3
"""
Remove incorrect long-distance CNT bonds from a GROMACS .itp file.

Inputs:
  1. NMA.itp
  2. box_ions.gro

The script:
  - Reads the CNT/TUBE atoms from the GRO file and ignores SOL.
  - Reads the [ bonds ] section of the ITP.
  - Uses minimum-image distances from the GRO box.
  - Removes bonds whose distance is greater than MAX_BOND_DISTANCE.
  - Removes [ pairs ] entries that become invalid because they are no
    longer a 1-4 path after the bad bond is removed.
  - Writes a new ITP and a report.

IMPORTANT:
  This script removes suspicious bonds; it does NOT automatically decide
  what the replacement bond should be. Inspect the report before MD.
"""

from pathlib import Path
import math
import re
from collections import defaultdict, deque

ITP_FILE = Path("NMA.itp")
GRO_FILE = Path("3_solv.gro")
OUTPUT_ITP = Path("NMA_corrected.itp")
REPORT_FILE = Path("NMA_correction_report.txt")

# CNT C-C bonds should be around 0.14 nm.
# 0.20 nm is deliberately conservative: anything above this is suspicious.
MAX_BOND_DISTANCE = 0.20

# Change to "TUBE" or "CNT" if you want strict residue filtering.
CNT_RESIDUES = {"TUBE", "CNT"}

def parse_gro(path):
    lines = path.read_text().splitlines()
    natoms = int(lines[1].strip())

    atoms = {}
    for line in lines[2:2 + natoms]:
        resid = int(line[0:5])
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        atomnr = int(line[15:20])
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        atoms[atomnr] = {
            "resid": resid,
            "resname": resname,
            "atomname": atomname,
            "x": x, "y": y, "z": z,
        }

    # Standard GRO box parsing. Supports triclinic boxes only partially;
    # for a normal orthorhombic CNT box this is sufficient.
    boxvals = [float(x) for x in lines[2 + natoms].split()]
    if len(boxvals) < 3:
        raise ValueError("Could not read GRO box dimensions.")
    box = boxvals[:3]

    return atoms, box

def minimum_image_delta(d, box_length):
    return d - box_length * round(d / box_length)

def distance_pbc(a, b, box):
    dx = minimum_image_delta(a["x"] - b["x"], box[0])
    dy = minimum_image_delta(a["y"] - b["y"], box[1])
    dz = minimum_image_delta(a["z"] - b["z"], box[2])
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def parse_itp_sections(path):
    lines = path.read_text().splitlines()
    sections = []
    current = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        m = re.match(r"^\[\s*([^\]]+)\s*\]$", stripped)
        if m:
            current = m.group(1).strip().lower()
        sections.append((i, current, line))

    return lines, sections

def data_tokens(line):
    # Remove comments before tokenizing.
    return line.split(";", 1)[0].split()

def build_bond_graph(bond_records):
    graph = defaultdict(set)
    for a, b, line_idx in bond_records:
        graph[a].add(b)
        graph[b].add(a)
    return graph

def has_three_bond_path(graph, start, target):
    """Return True if target is reachable from start in exactly 3 bonds."""
    frontier = {start}
    visited = {start}
    for depth in range(3):
        nxt = set()
        for node in frontier:
            for nb in graph[node]:
                if nb == target and depth == 2:
                    return True
                if nb not in visited:
                    visited.add(nb)
                    nxt.add(nb)
        frontier = nxt
    return False

def main():
    atoms, box = parse_gro(GRO_FILE)

    # Restrict analysis to CNT/TUBE atoms if such residues are present.
    cnt_atoms = {
        n: a for n, a in atoms.items()
        if a["resname"] in CNT_RESIDUES
    }

    if not cnt_atoms:
        raise RuntimeError(
            f"No atoms with residue names {sorted(CNT_RESIDUES)} found in {GRO_FILE}."
        )

    lines, sections = parse_itp_sections(ITP_FILE)

    bond_records = []
    pair_records = []

    for i, section, line in sections:
        if section == "bonds":
            tok = data_tokens(line)
            if len(tok) >= 2 and tok[0].isdigit() and tok[1].isdigit():
                bond_records.append((int(tok[0]), int(tok[1]), i))

        elif section == "pairs":
            tok = data_tokens(line)
            if len(tok) >= 2 and tok[0].isdigit() and tok[1].isdigit():
                pair_records.append((int(tok[0]), int(tok[1]), i))

    # Identify bad bonds using PBC distance.
    bad_bonds = []
    good_bonds = []

    for a, b, idx in bond_records:
        if a not in cnt_atoms or b not in cnt_atoms:
            good_bonds.append((a, b, idx))
            continue

        d = distance_pbc(cnt_atoms[a], cnt_atoms[b], box)

        if d > MAX_BOND_DISTANCE:
            bad_bonds.append((a, b, idx, d))
        else:
            good_bonds.append((a, b, idx))

    bad_bond_set = {
        frozenset((a, b)) for a, b, _, _ in bad_bonds
    }

    # Build graph after removing bad bonds.
    graph = defaultdict(set)
    for a, b, _ in good_bonds:
        graph[a].add(b)
        graph[b].add(a)

    # Remove a pair if it is no longer a 1-4 path in the corrected graph.
    # Keep pairs that are unrelated to removed bonds.
    bad_pairs = []
    keep_pairs = []

    for a, b, idx in pair_records:
        if has_three_bond_path(graph, a, b):
            keep_pairs.append((a, b, idx))
        else:
            bad_pairs.append((a, b, idx))

    # Write corrected ITP.
    remove_lines = {idx for _, _, idx, _ in bad_bonds}
    remove_lines.update(idx for _, _, idx in bad_pairs)

    with OUTPUT_ITP.open("w") as out:
        for i, line in enumerate(lines):
            if i not in remove_lines:
                out.write(line + "\n")

    # Report.
    with REPORT_FILE.open("w") as report:
        report.write("CNT topology correction report\n")
        report.write("=" * 60 + "\n\n")
        report.write(f"Input ITP: {ITP_FILE}\n")
        report.write(f"Input GRO: {GRO_FILE}\n")
        report.write(f"Output ITP: {OUTPUT_ITP}\n")
        report.write(f"Bond cutoff used: {MAX_BOND_DISTANCE:.3f} nm\n\n")

        report.write(f"Total CNT atoms found: {len(cnt_atoms)}\n")
        report.write(f"Total bonds examined: {len(bond_records)}\n")
        report.write(f"Bad bonds removed: {len(bad_bonds)}\n")
        report.write(f"Pairs removed: {len(bad_pairs)}\n\n")

        if bad_bonds:
            report.write("REMOVED BONDS\n")
            report.write("-" * 60 + "\n")
            for a, b, idx, d in bad_bonds:
                report.write(
                    f"ITP line {idx+1}: {a:5d} {b:5d}  "
                    f"distance = {d:.4f} nm  "
                    f"({cnt_atoms[a]['resname']}:{cnt_atoms[a]['atomname']} - "
                    f"{cnt_atoms[b]['resname']}:{cnt_atoms[b]['atomname']})\n"
                )
        else:
            report.write("No bonds exceeded the cutoff.\n")

        report.write("\nREMOVED PAIRS\n")
        report.write("-" * 60 + "\n")
        for a, b, idx in bad_pairs:
            report.write(f"ITP line {idx+1}: {a:5d} {b:5d}\n")

        report.write("\nIMPORTANT\n")
        report.write("-" * 60 + "\n")
        report.write(
            "This script removes suspicious long-distance bonds and pairs. "
            "It does NOT add replacement bonds. For a CNT, each carbon should "
            "normally have the appropriate neighboring carbon connectivity. "
            "Inspect the removed-bond list and add correct replacement bonds "
            "before production MD if necessary.\n"
        )

    print(f"Corrected ITP written to: {OUTPUT_ITP}")
    print(f"Report written to:         {REPORT_FILE}")
    print(f"Bad bonds removed:         {len(bad_bonds)}")
    print(f"Pairs removed:             {len(bad_pairs)}")

if __name__ == "__main__":
    main()
