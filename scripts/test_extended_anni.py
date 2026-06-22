"""Validate the single-photon+positron extension + flag gating (torch, no MEGAlib)."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from physics import physics_factors as pf


def test_single_photon_score():
    # hits 0+1 sum to 511 (the photon); hit2 = co-located (0.5 cm) positron deposit (150 keV)
    P = torch.tensor([[0.,0,0],[5,0,0],[0.5,0,0]])
    E = torch.tensor([300., 211., 150.])
    assert abs(float(pf._single_photon_positron_score(P, E)) - pf.SP_SCORE) < 1e-5, "should fire"
    # positron far away (10 cm) -> NaN
    Pf = P.clone(); Pf[2,0] = 10.0
    assert math.isnan(float(pf._single_photon_positron_score(Pf, E))), "far positron -> NaN"
    # positron too small (<60 keV) -> NaN
    Es = E.clone(); Es[2] = 30.0
    assert math.isnan(float(pf._single_photon_positron_score(P, Es))), "tiny deposit -> NaN"
    # no 511 subset -> NaN
    En = torch.tensor([100., 120., 150.])
    assert math.isnan(float(pf._single_photon_positron_score(P, En))), "no 511 -> NaN"
    print("  test_single_photon_score: PASS")

def test_flag_gating():
    # combo pool of one size-2 subset -> back_to_back is NaN (needs >=3 hits)
    pos = torch.tensor([[[0.,0,0],[5,0,0]]])      # (1, 2, 3)
    en = torch.tensor([[300., 211.]])             # (1, 2)
    sizes = torch.tensor([2])
    raw_p = torch.tensor([[0.,0,0],[5,0,0],[0.5,0,0]])
    raw_e = torch.tensor([300., 211., 150.])
    base = float(pf._back_to_back_score(pos, en, sizes=sizes))
    assert math.isnan(base), f"back_to_back should be NaN, got {base}"

    os.environ["BETADECAY_EXTENDED_ANNI"] = "0"
    off = pf.annihilation_angle(pos, sizes=sizes, energies=en, raw_positions=raw_p, raw_energies=raw_e)
    assert math.isnan(float(off)), "flag OFF -> raw ignored -> NaN (identical to champion)"

    os.environ["BETADECAY_EXTENDED_ANNI"] = "1"
    on = pf.annihilation_angle(pos, sizes=sizes, energies=en, raw_positions=raw_p, raw_energies=raw_e)
    assert abs(float(on) - pf.SP_SCORE) < 1e-5, f"flag ON -> single-photon fires, got {float(on)}"
    os.environ["BETADECAY_EXTENDED_ANNI"] = "0"
    print("  test_flag_gating: PASS (off=NaN like champion, on=SP_SCORE)")

def test_off_is_identical():
    # with flag off, passing raw hits must not change the back_to_back result
    pos = torch.tensor([[[0.,0,0],[5,0.1,0],[2.5,4,0]]])  # (1,3,3)
    en = torch.tensor([[260., 255., 10.]]); sizes = torch.tensor([3])
    raw_p = pos[0]; raw_e = en[0]
    os.environ["BETADECAY_EXTENDED_ANNI"] = "0"
    a = float(pf.annihilation_angle(pos, sizes=sizes, energies=en))
    b = float(pf.annihilation_angle(pos, sizes=sizes, energies=en, raw_positions=raw_p, raw_energies=raw_e))
    assert (math.isnan(a) and math.isnan(b)) or a == b, f"flag off must be identical: {a} vs {b}"
    print("  test_off_is_identical: PASS")

if __name__ == "__main__":
    test_single_photon_score(); test_flag_gating(); test_off_is_identical()
    print("ALL EXTENDED-ANNI CHECKS PASSED")
