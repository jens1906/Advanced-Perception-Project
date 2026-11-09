import zipfile
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader

from PreliminaryFunctions import FindSequences

# Download parameters
SOCCERNET_DIR = Path("Data/SoccerNet")
TRACKING_DIR = SOCCERNET_DIR / "tracking"
SPLITS = ["train", "test"]

if __name__ == "__main__":
    # Zips that are already downloaded are skipped by the downloader
    SoccerNetDownloader(LocalDirectory=str(SOCCERNET_DIR)).downloadDataTask(task="tracking", split=SPLITS)

    for split in SPLITS:
        splitDir = TRACKING_DIR / split
        if FindSequences(splitDir):
            print(f"{split}  already extracted to {splitDir}")
            continue
        # The zip holds a top-level <split>/ folder, so extract next to it
        with zipfile.ZipFile(TRACKING_DIR / f"{split}.zip") as archive:
            archive.extractall(TRACKING_DIR)
        print(f"{split}  {len(FindSequences(splitDir))} sequences extracted to {splitDir}")
