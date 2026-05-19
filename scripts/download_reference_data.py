"""Download public reference layers used by the riskscape workflow.

The workflow uses a small set of spatial reference layers for maps, grids, and
overlays. Natural Earth layers have stable public ZIP URLs. SAERI-hosted layers
are discovered through the public CKAN API when possible; if the portal changes
or a direct resource URL is unavailable, this script reports the source page and
expected local destination.

Usage examples:
    python scripts/download_reference_data.py
    python scripts/download_reference_data.py --dataset natural-earth
    python scripts/download_reference_data.py --dataset fisheries-grid conservation-zones
    python scripts/download_reference_data.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "reference"

CKAN_API_BASE = "https://dataportal.saeri.org/api/3/action/package_show"


@dataclass(frozen=True)
class DirectZipDataset:
    """Reference layer with a known static ZIP URL."""

    key: str
    label: str
    url: str
    destination: Path
    expected_file: str


@dataclass(frozen=True)
class CkanZipDataset:
    """Reference layer discovered from a CKAN package."""

    key: str
    label: str
    package_ids: tuple[str, ...]
    destination: Path
    expected_file: str
    source_page: str
    resource_name_hints: tuple[str, ...]


NATURAL_EARTH_DATASETS = (
    DirectZipDataset(
        key="ne-land",
        label="Natural Earth 10m land",
        url="https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip",
        destination=REFERENCE_DIR / "ne_10m_land",
        expected_file="ne_10m_land.shp",
    ),
    DirectZipDataset(
        key="ne-coastline",
        label="Natural Earth 10m coastline",
        url=(
            "https://naturalearth.s3.amazonaws.com/10m_physical/"
            "ne_10m_coastline.zip"
        ),
        destination=REFERENCE_DIR / "ne_10m_coastline",
        expected_file="ne_10m_coastline.shp",
    ),
)

SAERI_DATASETS = (
    CkanZipDataset(
        key="fisheries-grid",
        label="Falkland Islands fisheries grid squares",
        package_ids=("falkland-islands-fisheries-grid-squares",),
        destination=REFERENCE_DIR / "fisheries_grid_squares",
        expected_file="GridSquares.shp",
        source_page=(
            "https://dataportal.saeri.org/dataset/"
            "falkland-islands-fisheries-grid-squares"
        ),
        resource_name_hints=("fisheries_grid_squares", "grid"),
    ),
    CkanZipDataset(
        key="conservation-zones",
        label="Falkland Islands FICZ/FOCZ conservation zones",
        package_ids=(
            "falkland-islands-interim-conservation-zone-ficz-and-"
            "falkland-islands-outer-conservation-zone-focz",
            "falkland-islands-conservation-zones",
        ),
        destination=REFERENCE_DIR / "ukho_ficz_focz_limits",
        expected_file="ukho_ficz_focz_limits.shp",
        source_page=(
            "https://dataportal.saeri.org/dataset/"
            "falkland-islands-conservation-zones"
        ),
        resource_name_hints=("ukho_ficz_focz_limits", "conservation", "ficz"),
    ),
)

DATASETS = {
    "natural-earth": NATURAL_EARTH_DATASETS,
    **{dataset.key: (dataset,) for dataset in NATURAL_EARTH_DATASETS},
    **{dataset.key: (dataset,) for dataset in SAERI_DATASETS},
}


def download_url(url: str, destination: Path) -> None:
    """Download a URL to a local path."""
    request = urllib.request.Request(url, headers={"User-Agent": "riskscape/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def extract_zip(zip_path: Path, destination: Path, overwrite: bool) -> None:
    """Extract a ZIP archive into a destination folder."""
    if overwrite and destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def expected_file_exists(dataset: DirectZipDataset | CkanZipDataset) -> bool:
    """Return whether the expected reference file is already present."""
    return (dataset.destination / dataset.expected_file).exists()


def download_direct_zip(dataset: DirectZipDataset, overwrite: bool) -> bool:
    """Download and extract a dataset with a static ZIP URL."""
    if expected_file_exists(dataset) and not overwrite:
        print(f"[OK] {dataset.label} already available")
        return True

    print(f"[download] {dataset.label}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / f"{dataset.key}.zip"
        download_url(dataset.url, zip_path)
        extract_zip(zip_path, dataset.destination, overwrite=overwrite)

    if expected_file_exists(dataset):
        print(f"[OK] {dataset.label} -> {dataset.destination}")
        return True

    print(f"[error] Downloaded {dataset.label}, but {dataset.expected_file} was not found")
    return False


def load_ckan_package(package_id: str) -> dict:
    """Load CKAN package metadata."""
    url = f"{CKAN_API_BASE}?id={package_id}"
    request = urllib.request.Request(url, headers={"User-Agent": "riskscape/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not payload.get("success"):
        raise RuntimeError(f"CKAN package lookup failed for {package_id}")

    return payload["result"]


def score_resource(resource: dict, hints: Iterable[str]) -> int:
    """Score a CKAN resource by how likely it is to be the desired ZIP."""
    name = str(resource.get("name") or "").lower()
    url = str(resource.get("url") or "").lower()
    fmt = str(resource.get("format") or "").lower()
    combined = f"{name} {url} {fmt}"

    score = 0
    if "zip" in fmt or url.endswith(".zip"):
        score += 10
    if "geojson" in fmt or url.endswith(".geojson"):
        score -= 5
    for hint in hints:
        if hint.lower() in combined:
            score += 3

    return score


def find_ckan_zip_url(dataset: CkanZipDataset) -> str | None:
    """Find a likely ZIP resource URL for a CKAN-backed dataset."""
    for package_id in dataset.package_ids:
        try:
            package = load_ckan_package(package_id)
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            print(f"[warn] Could not query SAERI package {package_id}: {exc}")
            continue

        resources = package.get("resources", [])
        ranked = sorted(
            resources,
            key=lambda resource: score_resource(
                resource,
                dataset.resource_name_hints,
            ),
            reverse=True,
        )

        for resource in ranked:
            resource_url = resource.get("url")
            if resource_url and score_resource(resource, dataset.resource_name_hints) >= 10:
                return str(resource_url)

    return None


def print_manual_instructions(dataset: CkanZipDataset) -> None:
    """Print manual fallback instructions for a reference layer."""
    print(f"[manual] {dataset.label} could not be downloaded automatically.")
    print(f"         Source: {dataset.source_page}")
    print(f"         Destination: {dataset.destination.relative_to(PROJECT_ROOT)}")
    print(f"         Expected file: {dataset.expected_file}")
    print("         Keep the shapefile sidecar files together (.shp, .shx, .dbf, .prj).")


def download_ckan_zip(dataset: CkanZipDataset, overwrite: bool) -> bool:
    """Download and extract a CKAN-backed reference dataset."""
    if expected_file_exists(dataset) and not overwrite:
        print(f"[OK] {dataset.label} already available")
        return True

    url = find_ckan_zip_url(dataset)
    if not url:
        print_manual_instructions(dataset)
        return False

    print(f"[download] {dataset.label}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / f"{dataset.key}.zip"
        download_url(url, zip_path)
        extract_zip(zip_path, dataset.destination, overwrite=overwrite)

    if expected_file_exists(dataset):
        print(f"[OK] {dataset.label} -> {dataset.destination}")
        return True

    print(f"[warn] Downloaded {dataset.label}, but {dataset.expected_file} was not found")
    print_manual_instructions(dataset)
    return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download public reference layers for the riskscape workflow",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=sorted(DATASETS),
        help="Reference dataset(s) to download. Defaults to all datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing reference folders before extracting downloads.",
    )
    return parser.parse_args()


def selected_datasets(names: list[str] | None) -> list[DirectZipDataset | CkanZipDataset]:
    """Resolve command-line dataset groups to concrete datasets."""
    if not names:
        return [*NATURAL_EARTH_DATASETS, *SAERI_DATASETS]

    selected: list[DirectZipDataset | CkanZipDataset] = []
    seen: set[str] = set()
    for name in names:
        for dataset in DATASETS[name]:
            if dataset.key not in seen:
                selected.append(dataset)
                seen.add(dataset.key)

    return selected


def main() -> int:
    """Download selected reference datasets."""
    args = parse_args()

    success_count = 0
    datasets = selected_datasets(args.dataset)

    for dataset in datasets:
        try:
            if isinstance(dataset, DirectZipDataset):
                success = download_direct_zip(dataset, overwrite=args.overwrite)
            else:
                success = download_ckan_zip(dataset, overwrite=args.overwrite)
        except (OSError, urllib.error.URLError, zipfile.BadZipFile) as exc:
            print(f"[error] {dataset.label}: {exc}")
            success = False

        success_count += int(success)

    print(f"[summary] {success_count}/{len(datasets)} reference datasets ready")
    return 0 if success_count == len(datasets) else 1


if __name__ == "__main__":
    sys.exit(main())
