"""One-off script to fetch the Casting Product defect dataset and place it
where the pipeline expects it. Not part of the production app — run once."""
import shutil
import kagglehub
from pathlib import Path

path = kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product")
print("Downloaded to:", path)

dest = Path(__file__).resolve().parents[1] / "data" / "raw"
dest.mkdir(parents=True, exist_ok=True)

# Copy the whole extracted folder structure into data/raw
src = Path(path)
for item in src.iterdir():
    target = dest / item.name
    if item.is_dir():
        shutil.copytree(item, target, dirs_exist_ok=True)
    else:
        shutil.copy(item, target)

print("Copied dataset into:", dest)