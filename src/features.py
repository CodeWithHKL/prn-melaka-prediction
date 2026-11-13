"""Build the per-seat feature table for Melaka's 28 state (DUN) seats.

Input : raw Tindak Malaysia CSVs in data/raw/tindak/ (see fetch_data.py).
Output: data/processed/seats_long.csv      one row per seat per election (2013, 2018, 2021)
        data/processed/seats_features.csv  one row per seat, wide format, used by the model

Vote shares are expressed as a fraction of valid votes for three blocs:
    BN   Barisan Nasional
    PH   Pakatan Harapan (in 2013: PKR / DAP, i.e. the opposition "PR" minus PAS)
    PN   Perikatan Nasional (in 2013: PAS; in 2018: Gagasan Sejahtera "GS")
    OTH  everything else (independents, minor parties)
Bersatu counted as part of PN in 2021.
"""
import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/tindak")
OUT = Path("data/processed")


# --------------------------------------------------------------------------- helpers
def to_number(s):
    """Parse numbers that may contain thousands separators ("10,033") or be blank."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0)


def seat_number(code):
    """'N. 01' / 'N.01' -> 1."""
    return int(re.search(r"(\d+)", str(code)).group(1))


def read_raw(filename):
    # utf-8-sig strips the byte-order mark some of these files start with
    return pd.read_csv(RAW / filename, encoding="utf-8-sig")


def is_melaka(df):
    return df["STATE"].str.upper().str.contains("MELAKA|MALACCA")


def make_shares(df, year, bn_votes, ph_votes, pn_votes):
    """Turn raw vote counts into one tidy row per seat with bloc vote shares."""
    valid = to_number(df["TOTAL VALID VOTES"])
    out = pd.DataFrame({
        "seat": df["STATE CONSTITUENCY CODE"].map(seat_number),
        "name": df["STATE CONSTITUENCY NAME"].str.strip().str.upper(),
        "year": year,
        "electorate": to_number(df["TOTAL ELECTORATE"]),
        "turnout": to_number(df["TURNOUT (%)"]),
        "valid": valid,
    })
    out["BN"] = bn_votes / valid
    out["PH"] = ph_votes / valid
    out["PN"] = pn_votes / valid
    out["OTH"] = (1 - out["BN"] - out["PH"] - out["PN"]).clip(lower=0)
    return out


# --------------------------------------------------------------------------- one builder per election
def build_2013():
    """2013 lists the opposition as PR (1) / PR (2); split them by party into PH vs PAS."""
    d = read_raw("melaka_2013_dun.csv")
    ph = pd.Series(0.0, index=d.index)
    pn = pd.Series(0.0, index=d.index)
    for slot in ("PR (1)", "PR (2)"):
        votes = to_number(d[f"{slot} VOTE"])
        is_pas = d[slot].astype(str).str.upper().str.contains("PAS")
        pn += votes.where(is_pas, 0)
        ph += votes.where(~is_pas, 0)
    return make_shares(d, 2013, to_number(d["BN VOTE"]), ph, pn)


def build_2018():
    d = read_raw("my_2018_dun.csv")
    d = d[is_melaka(d)].reset_index(drop=True)  # file covers every state
    return make_shares(d, 2018, to_number(d["BN VOTE"]), to_number(d["PH VOTE"]), to_number(d["GS VOTE"]))


def build_2021():
    d = read_raw("melaka_2021_results.csv")
    return make_shares(d, 2021, to_number(d["BN VOTE"]), to_number(d["PH VOTE"]),
                       to_number(d["PN CANDIDATE VOTE"]))  # column is oddly named in the source


def build_ge15(seat_to_parliament):
    """GE15 (Nov 2022) results exist only per parliament seat (6 in Melaka).

    Each state seat inherits the share of the parliament seat it sits inside, so
    this is a coarse regional signal, not a per-seat one.
    """
    p = read_raw("my_2022_parlimen.csv")
    p = p[is_melaka(p)]
    valid = to_number(p["TOTAL VALID VOTES"])
    ge = pd.DataFrame({
        "par": p["PARLIAMENTARY CONSTITUENCY NAME"].str.strip().str.upper(),
        "ge15_BN": to_number(p["BN VOTE"]) / valid,
        "ge15_PH": to_number(p["PH VOTE"]) / valid,
        "ge15_PN": to_number(p["PN VOTE"]) / valid,
    })
    ge["ge15_OTH"] = (1 - ge[["ge15_BN", "ge15_PH", "ge15_PN"]].sum(axis=1)).clip(lower=0)
    return seat_to_parliament.merge(ge, on="par", how="left")


def build_demographics():
    """2021 voter composition per state seat (ethnic mix, age, urban/rural)."""
    c = read_raw("melaka_2021_composition.csv")
    demo = pd.DataFrame({
        "seat": c["STATE CONSTITUENCY CODE"].map(seat_number),
        "par": c["PARLIAMENTARY CONSTITUENCY NAME"].str.strip().str.upper(),
        "malay": c["MALAY (%)"],
        "chinese": c["CHINESE (%)"],
        "indian": c["INDIANS (%)"],
        "age21_29": c["21 - 29 (%)"],
        "age30_39": c["30 - 39 (%)"],
    })
    urban_label = c["URBAN-RURAL CLASSIFICATION (2021)"].astype(str).str.upper()
    demo["urban"] = urban_label.str.contains("URBAN").astype(int)
    return demo


# --------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    long = pd.concat([build_2013(), build_2018(), build_2021()], ignore_index=True)
    demo = build_demographics()
    ge15 = build_ge15(demo[["seat", "par"]])

    # long -> wide: one column per bloc per year, e.g. BN_2021
    wide = long.pivot(index="seat", columns="year", values=["BN", "PH", "PN", "OTH"])
    wide.columns = [f"{bloc}_{year}" for bloc, year in wide.columns]
    wide = wide.reset_index()

    seats_2021 = long[long.year == 2021][["seat", "name", "electorate"]]
    feats = (seats_2021
             .merge(wide, on="seat")
             .merge(demo.drop(columns="par"), on="seat")
             .merge(ge15.drop(columns="par"), on="seat"))

    long.to_csv(OUT / "seats_long.csv", index=False)
    feats.to_csv(OUT / "seats_features.csv", index=False)

    # sanity checks: fail loudly if a source file changed shape
    assert feats["seat"].nunique() == 28, "expected 28 seats"
    assert feats.notna().all().all(), feats.isna().sum()[lambda s: s > 0]
    print(f"OK: {len(feats)} seats, {feats.shape[1]} columns -> {OUT / 'seats_features.csv'}")


if __name__ == "__main__":
    sys.exit(main())
