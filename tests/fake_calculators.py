"""Tiny deterministic ASE calculator used to exercise the optional factory seam."""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        # A smooth, deterministic toy surface.  Its only purpose is to prove
        # that module:function factories are loaded and evaluated consistently;
        # it is not a chemistry model.
        self.results["energy"] = float(np.sum(positions[:, 0] ** 2))
        forces = np.zeros_like(positions)
        forces[:, 0] = -2.0 * positions[:, 0]
        self.results["forces"] = forces


def make_calculator(**kwargs):
    del kwargs
    return HarmonicCalculator()
