"""NASA OceanColor chlorophyll downloader."""

from pathlib import Path
import subprocess

from riskscape.config import cfg


SEARCH_URL = "https://oceandata.sci.gsfc.nasa.gov/file_search/"


def download(dataset_cfg: dict, output_dir: Path) -> None:
    """Download VIIRS L3 chlorophyll files using wget."""

    start = cfg["time"]["start"] + " 00:00:00"
    end = cfg["time"]["end"] + " 23:59:59"

    payload = (
        f"results_as_file=1&sensor_id=7"
        f"&sdate={start}&edate={end}&subType=1"
    )

    list_file = output_dir / "file_list.txt"

    print("Querying OceanColor file list")

    with open(list_file, "w") as f:
        subprocess.run(
            [
                "wget",
                "-q",
                "--post-data",
                payload,
                "-O",
                "-",
                SEARCH_URL,
            ],
            stdout=f,
            check=True,
        )

    print("Downloading files")

    subprocess.run(
        [
            "wget",
            "--continue",
            "--input-file",
            str(list_file),
            "--directory-prefix",
            str(output_dir),
        ],
        check=True,
    )