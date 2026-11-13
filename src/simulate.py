"""Monte Carlo seat simulation for the Melaka 2026 state election (PRN).

Four blocs compete:
    PACT     BN + official PN (PAS / Gerakan / Wawasan). They never contest the same
             seat, so the pact fields exactly one candidate everywhere.
    PH       Pakatan Harapan
    BSTU     Bersatu + Putra, which are OUTSIDE the pact
    BERSAMA  new party contesting alone

Each simulation run:
    1. Start from 2021 vote shares, nudged toward GE15 (GE15_WEIGHT).
    2. Draw state-wide parameters: BN<->PN vote transfer rate, and the share of
       2021 "PN" voters that follows Bersatu/Putra where they stand.
    3. Build each bloc's share per seat from those, add a state-wide swing and
       seat-level noise, and keep only blocs that actually have a candidate.
    4. Bersama takes a share where it stands; independents take a flat 1%.
    5. Highest share wins the seat.
Repeat N_SIMS times, then count seats.

Every number in PARAMS is a judgement call, not something fitted to data
(the backtest in model.py was too weak to fit anything). See README.md.

Input : data/processed/seats_features.csv, data/processed/lineup_2026.csv
Output: data/processed/seat_probabilities_2026.csv, seat_totals_sims_2026.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/processed")
N_SIMS = 20_000
SEED = 2026
N_SEATS = 28
MAJORITY = 15

PARAMS = dict(
    ge15_weight=0.4,       # fraction of the 2021 -> GE15 shift carried into 2026
    state_sd=0.06,         # state-wide swing SD per bloc (same shift in every seat)
    seat_sd=0.09,          # seat-level noise SD (backtest errors were 11-17pp)
    transfer_mean=0.80,    # BN <-> PN vote transfer within the pact
    transfer_sd=0.08,
    bstu_keep_mean=0.50,   # share of 2021 PN voters following Bersatu/Putra where they stand
    bstu_keep_sd=0.15,
    bersama_mean=0.08,     # Bersama's baseline share where it stands
    bersama_sd=0.04,
    bersama_leak=0.5,      # share of votes the pact fails to transfer that Bersama picks up
    bebas_share=0.01,      # flat share for an independent
)

BLOCS = ["PACT", "PH", "BSTU"]          # blocs tracked as vote shares
WINNERS = BLOCS + ["BERSAMA"]           # blocs that can win (index 3 = Bersama)

# Candidate counts reported at nomination; used to catch typos in lineup_2026.csv
EXPECTED_CANDIDATES = {"BN": 23, "PH": 24, "Bersama": 16, "Bersatu": 7, "PN": 5, "Putra": 1, "Bebas": 3}


# --------------------------------------------------------------------------- inputs
def bloc_of(party):
    """Map a party label from the lineup file to one of the simulation blocs."""
    p = party.upper()
    if p.startswith(("BN", "PN")):
        return "PACT"
    if p.startswith("PH"):
        return "PH"
    if p in ("BERSATU", "PUTRA"):
        return "BSTU"
    return "BERSAMA" if p == "BERSAMA" else "BEBAS"


def load_lineup():
    """Read the 2026 candidate list and work out which blocs stand in which seat."""
    lineup = pd.read_csv(OUT / "lineup_2026.csv")
    lineup["bloc"] = lineup["party"].map(bloc_of)

    counts = lineup["party"].str.split("-").str[0].value_counts().to_dict()
    assert len(lineup) == 79 and lineup["seat"].nunique() == N_SEATS, "expected 79 candidates over 28 seats"
    assert counts == EXPECTED_CANDIDATES, (counts, EXPECTED_CANDIDATES)

    # boolean array per bloc: does it field a candidate in seat i?
    stands = {b: np.zeros(N_SEATS, bool) for b in WINNERS + ["BEBAS", "BN"]}
    for _, row in lineup.iterrows():
        stands[row["bloc"]][row["seat"] - 1] = True
        if row["party"].upper().startswith("BN"):
            stands["BN"][row["seat"] - 1] = True  # BN specifically (vs PN) decides who is the "own" side

    pn_seats = lineup[lineup.party.str.upper().str.startswith("PN")]["seat"].values - 1
    assert not stands["BN"][pn_seats].any(), "BN and PN must never contest the same seat"
    assert stands["PACT"].all(), "every seat needs a pact candidate"
    return lineup, stands


def load_base_shares(feats):
    """Expected 2026 starting shares per seat as columns (BN, PH, PN): 2021 nudged toward GE15."""
    def normalised(cols):
        m = np.column_stack([feats[c] for c in cols])
        return m / m.sum(axis=1, keepdims=True)

    s2021 = normalised(["BN_2021", "PH_2021", "PN_2021"])
    ge15 = normalised(["ge15_BN", "ge15_PH", "ge15_PN"])
    mu = s2021 + PARAMS["ge15_weight"] * (ge15 - s2021)
    return np.clip(mu, 0.005, None)


# --------------------------------------------------------------------------- one simulation
def simulate_once(rng, mu, stands):
    """Return (shares[28, 3] for PACT/PH/BSTU, winner index per seat)."""
    bn, ph, pn = mu[:, 0], mu[:, 1], mu[:, 2]
    has_bn, has_bstu = stands["BN"], stands["BSTU"]

    # state-wide parameters, drawn once per run (so they act like a shared swing)
    transfer = np.clip(rng.normal(PARAMS["transfer_mean"], PARAMS["transfer_sd"]), 0.4, 1.0)
    keep = np.clip(rng.normal(PARAMS["bstu_keep_mean"], PARAMS["bstu_keep_sd"]), 0.1, 0.9)
    keep = np.where(has_bstu, keep, 0.0)  # PN voters only defect where Bersatu/Putra stands

    # pact vote: its own side's voters + a transferred slice of the other side's
    pn_staying = (1 - keep) * pn
    pact = np.where(has_bn, bn + transfer * pn_staying,   # BN candidate: BN + transferred PN
                    pn_staying + transfer * bn)           # PN candidate: PN + transferred BN
    bstu = keep * pn
    leaked = (1 - transfer) * np.where(has_bn, pn_staying, bn)  # votes the pact fails to carry

    # swing + noise; absent blocs get no votes; rescale so the three blocs sum to 1
    s = np.column_stack([pact, ph, bstu])
    s = s + rng.normal(0, PARAMS["state_sd"], 3) + rng.normal(0, PARAMS["seat_sd"], (N_SEATS, 3))
    s = np.clip(s, 0.0, None)
    s[:, 2] *= has_bstu
    s = s / s.sum(axis=1, keepdims=True)

    # Bersama and independents take their cut off the top
    mean_b = PARAMS["bersama_mean"] + PARAMS["bersama_leak"] * leaked
    bersama = np.where(stands["BERSAMA"], np.clip(rng.normal(mean_b, PARAMS["bersama_sd"]), 0.01, 0.6), 0)
    bebas = np.where(stands["BEBAS"], PARAMS["bebas_share"], 0)
    s = s * (1 - bersama - bebas)[:, None]

    winner = np.where(bersama > s.max(axis=1), 3, s.argmax(axis=1))
    return s, winner


# --------------------------------------------------------------------------- main
def main():
    feats = pd.read_csv(OUT / "seats_features.csv").sort_values("seat").reset_index(drop=True)
    lineup, stands = load_lineup()
    mu = load_base_shares(feats)

    rng = np.random.default_rng(SEED)
    winners = np.zeros((N_SIMS, N_SEATS), int)
    shares = np.zeros((N_SIMS, N_SEATS, 3))
    for i in range(N_SIMS):
        shares[i], winners[i] = simulate_once(rng, mu, stands)

    # per-seat results
    prob = pd.DataFrame({b: (winners == k).mean(axis=0) for k, b in enumerate(WINNERS)})
    prob.insert(0, "name", feats["name"])
    prob.insert(0, "seat", feats["seat"])
    for k, b in enumerate(BLOCS):
        prob[f"share_{b}"] = shares[:, :, k].mean(axis=0)
    prob["contest"] = [" / ".join(lineup[lineup.seat == s]["party"]) for s in feats["seat"]]
    prob.round(3).to_csv(OUT / "seat_probabilities_2026.csv", index=False)

    # seat totals per run
    totals = pd.DataFrame({b: (winners == k).sum(axis=1) for k, b in enumerate(WINNERS)})
    assert (totals.sum(axis=1) == N_SEATS).all(), "seat totals must sum to 28"
    totals.to_csv(OUT / "seat_totals_sims_2026.csv", index=False)

    for b in WINNERS:
        print(f"{b:8s} mean {totals[b].mean():5.1f}  80% range [{int(totals[b].quantile(.1))}-"
              f"{int(totals[b].quantile(.9))}]  P(>={MAJORITY}) {(totals[b] >= MAJORITY).mean():.3f}")
    print(f"P(no bloc reaches {MAJORITY}): {(totals.max(axis=1) < MAJORITY).mean():.3f}")


if __name__ == "__main__":
    sys.exit(main())
