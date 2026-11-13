"""Download the raw election CSVs used by the pipeline.

Source: Tindak Malaysia's HISTORICAL-ELECTION-RESULTS repository on GitHub
(https://github.com/TindakMalaysia/HISTORICAL-ELECTION-RESULTS).
Check the repository for its licence and attribution terms before redistributing.

Files are saved to data/raw/tindak/ under short local names, which features.py expects.
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/TindakMalaysia/HISTORICAL-ELECTION-RESULTS/master"
DEST = Path("data/raw/tindak")

# local file name -> path inside the repository
FILES = {
    "melaka_2013_dun.csv": "2013-ELECTION-RESULTS/ARCHIVED_FILES/MELAKA_2013_DUN_RESULTS.csv",
    "my_2018_dun.csv": "2018-ELECTION-RESULTS/MALAYSIA_2018_DUN_RESULTS.csv",  # all states; filtered later
    "melaka_2021_results.csv": "2021-MELAKA-STATE-ELECTIONS/MELAKA_2021_ELECTION_RESULTS.csv",
    "melaka_2021_composition.csv": "2021-MELAKA-STATE-ELECTIONS/MELAKA_2021_DUN_COMPOSITION.csv",
    "my_2022_parlimen.csv": "2022-ELECTION-RESULTS/MALAYSIA_2022_PARLIAMENT_RESULTS.csv",  # GE15, all states
}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for local, remote in FILES.items():
        url = f"{BASE}/{remote}"
        print(f"downloading {local} ...")
        urllib.request.urlretrieve(url, DEST / local)
    print(f"done: {len(FILES)} files in {DEST}")


if __name__ == "__main__":
    sys.exit(main())
