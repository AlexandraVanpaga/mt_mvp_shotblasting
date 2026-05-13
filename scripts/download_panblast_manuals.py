"""Download all PanBlast PDF manuals into the local `data/` folder.

The PanBlast website (https://www.panblast.com/manuals.acv) lists manuals
grouped into 11 categories. Each category page exposes a number of product
links, several of which point at the same underlying PDF (the PDF lives at
`https://www.panblast.com/xchange/manuals/<file>.pdf`).

This script:
  * Downloads every unique PDF found across all category pages.
  * Organises files into `data/<category>/<pdf_filename>`.
  * Writes a `data/manifest.json` mapping every product code to its PDF.
  * Is idempotent: existing PDFs of the right size are skipped.

Usage (from the project root):

    python scripts/download_panblast_manuals.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_PDF_URL = "https://www.panblast.com/xchange/manuals/"

# Mapping: category folder name -> list of (product_code_with_description, pdf_filename)
MANUALS: dict[str, list[tuple[str, str]]] = {
    "abrasive_metering_valves": [
        ("BAC-VA-PB-0060 Fina Valve 1-1/4 M X 1-1/4 M Assembly", "ZVP-PC-0071-01.pdf"),
        ("BAC-VA-PB-0061 Fina Valve 1-1/2 M X 1-1/2 M Assembly", "ZVP-PC-0071-01.pdf"),
        ("BAC-VA-PB-0062 Fina Valve 1-1/4 M X 1-1/2 M Assembly", "ZVP-PC-0071-01.pdf"),
        ("BAC-VA-PB-0063 Fina Valve 1 F X 1-1/2 M Assembly", "ZVP-PC-0071-01.pdf"),
        ("BAC-VA-PB-0089 Corsa II 1-1/4 Tungsten Sleeve Valve", "ZVP-PC-0086-00.pdf"),
        ("BAC-VA-PB-0114 Corsa II 1-1/2 Tungsten Sleeve Valve", "ZVP-PC-0086-00.pdf"),
        ("BAC-VA-0335-00 Junior Plana NPT Valve", "ZVP-PC-0110-00.pdf"),
        ("BAC-VA-0353-00 AbraFlo NPT Abrasive Trap II Assembly", "ZVP-PC-0111-00.pdf"),
    ],
    "abrasive_recycling_systems": [
        ("BEC-RV-PB-0001 Pow Air Vac Assembly", "ZVP-PC-0100-00.pdf"),
        ("BAC-RV-PB-0018 Pow Air Vac Drum Clamp Kit", "ZVP-PC-0100-00.pdf"),
    ],
    "pressure_blast_machine_consumables": [
        ("BAC-NZ-PB-0001 Nozzle Blast UTN-4 Urethane Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0002 Nozzle Blast UTN-5 Urethane Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0003 Nozzle Blast UTN-6 Urethane Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0004 Nozzle Blast UTN-7 Urethane Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0005 Nozzle Blast UTN-8 Urethane Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0006 Nozzle Blast ATN-4 Aluminium Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0007 Nozzle Blast ATN-5 Aluminium Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0008 Nozzle Blast ATN-6 Aluminium Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0009 Nozzle Blast ATN-7 Aluminium Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0010 Nozzle Blast ATN-8 Aluminium Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0011 Nozzle Blast UBN-4 Urethane Boron", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0012 Nozzle Blast UBN-5 Urethane Boron", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0013 Nozzle Blast UBN-6 Urethane Boron", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0014 Nozzle Blast UBN-7 Urethane Boron", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0015 Nozzle Blast UBN-8 Urethane Boron", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0016 Nozzle Blast ABN-4 Aluminium Boron", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0017 Nozzle Blast ABN-5 Aluminium Boron", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0018 Nozzle Blast ABN-6 Aluminium Boron", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0019 Nozzle Blast ABN-7 Aluminium Boron", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0020 Nozzle Blast ABN-8 Aluminium Boron", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0053 Nozzle Blast ATN-2S Alum.Tungsten Short", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0054 Nozzle Blast ATN-3S Alum.Tungsten Short", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0055 Nozzle Blast ATN-4S Alum.Tungsten Short", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0056 Nozzle Blast ATN-5S Alum.Tungsten Short", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0057 Nozzle Blast ATN-6S Alum.Tungsten Short", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0058 Nozzle Blast ATN-8S Alum.Tungsten Short", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0059 Nozzle Blast ABN-2S Alum.Boron Short", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0060 Nozzle Blast ABN-3S Alum.Boron Short", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0061 Nozzle Blast ABN-4S Alum.Boron Short", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0062 Nozzle Blast ABN-5S Alum.Boron Short", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0063 Nozzle Blast ABN-6S Alum.Boron Short", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0064 Nozzle Blast ABN-8S Alum.Boron Short", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0181 Nozzle Blast UTN-3 Urethane Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0185 Nozzle Blast UTN-10 Urethane Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0279 Nozzle Blast ATN-3M Alum.Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0280 Nozzle Blast ATN-4M Alum.Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0362 No.5 Alumin/Tung Double Venturi Nozzle", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0363 No.6 Alumin/Tung Double Venturi Nozzle", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0364 No.7 Alumin/Tung Double Venturi Nozzle", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-PB-0365 No.8 Alumin/Tung Double Venturi Nozzle", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-0478-00 Nozzle Blast ATN-5T Alum.Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-0478-01 Nozzle Blast ATN-5T Alum.Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-0482-00 Nozzle Blast ATN-6T Alum.Tungsten", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-0482-01 Nozzle Blast ATN-6T Alum.Tungsten", "ZVP-PC-0027-01.pdf"),
        ("YAC-NZ-0544-00 Boron Nozzle-Custom Flat Pattern", "ZVP-PC-0027-01.pdf"),
        ("BAC-NZ-0558-00 Nozzle Blast Nylon Silicon NSN-6", "ZVP-PC-0027-01.pdf"),
    ],
    "remote_control_valves_and_handles": [
        ("BAC-RC-PB-0014 Uniflo Valve Assembly With Fittings", "ZVP-PC-0072-01.pdf"),
        ("BAC-RC-PB-0085 AirFlo Pneumatic Control Handle", "ZVP-PC-0061-01.pdf"),
        ("BAC-RC-PB-0139 AirFlo NPT Control Handle", "ZVP-PC-0061-01.pdf"),
        ("BAC-RC-0494-00 AirFlo Control Handle With JIC Fitting", "ZVP-PC-0061-01.pdf"),
        ("BAC-RC-0525-00 AirFlo Control Handle With Unres JIC Fitting", "ZVP-PC-0061-01.pdf"),
        ("BAC-RC-0549-00 AirStop IV NPT Pneumatic Control Handle", "ZVP-PC-0091-00.pdf"),
    ],
    "supplied_air_respirator_helmets_and_accessories": [
        ("BAC-AF-PB-0032 Air Cooling Controller And Belt Assembly", "ZVP-PC-0042-01.pdf"),
        ("BAC-AF-PB-0036 Air Flow Controller And Belt Assembly", "ZVP-PC-0039-01.pdf"),
        ("BAC-AF-PB-0175 Climate Controller And Belt Assembly", "ZVP-PC-0043-01.pdf"),
        ("BAC-BH-PB-0004 Standard Supplied Air Respirator Helmet Cape Assembly", "ZVP-PC-0038-01.pdf"),
        ("BAC-BH-0022-05 Titan II Respirator Helmet With Standard Cape (AS/NZS)", "ZVP-PC-0055-00.pdf"),
        ("BAC-BH-PB-0037 Spartan SAR With Complete Cape", "ZVP-PC-0005-02.pdf"),
        ("BAC-BH-PB-0076 Spartan SAR With Air Flow Controller", "ZVP-PC-0005-02.pdf"),
        ("BAC-BH-0138-00 Titan II Respirator Helmet With Standard Cape & Air Flow Controller (AS/NZS)", "ZVP-PC-0055-00.pdf"),
        ("BAC-BH-0139-00 Titan II Respirator Helmet With Leather Cape & Air Flow Controller (AS/NZS)", "ZVP-PC-0055-00.pdf"),
        ("BAC-BH-0141-00 Titan II Respirator Helmet With Standard Cape & Air Cooling Controller (AS/NZS)", "ZVP-PC-0055-00.pdf"),
        ("BAC-BH-0142-00 Titan II Respirator Helmet With Leather Cape & Air Cooling Controller (AS/NZS)", "ZVP-PC-0055-00.pdf"),
        ("BAC-BH-0144-00 Titan II Respirator Helmet With Standard Cape & Climate Controller (AS/NZS)", "ZVP-PC-0055-00.pdf"),
        ("BAC-BH-0145-00 Titan II Respirator Helmet With Leather Cape & Climate Controller (AS/NZS)", "ZVP-PC-0055-00.pdf"),
        ("BAC-BH-0161-03 Breathing Tube Assembly", "ZVP-PC-0041-01.pdf"),
        ("BAC-BH-0175-00 Cosmo SAR & Air Flow Controller (CE)", "ZVP-PC-0078-01.pdf"),
        ("BAC-BH-0176-00 Cosmo SAR & Air Cooling Controller (CE)", "ZVP-PC-0078-01.pdf"),
        ("BAC-BH-0177-00 Cosmo SAR & Climate Controller (CE)", "ZVP-PC-0078-01.pdf"),
        ("BAC-BH-0178-00 Cosmo SAR With Cape & Breathing Tube (CE)", "ZVP-PC-0078-01.pdf"),
        ("BAC-BH-0180-00 Galaxy SAR & Air Flow Controller", "ZVP-PC-0079-00.pdf"),
        ("BAC-BH-0181-00 Galaxy SAR & Air Cooling Controller", "ZVP-PC-0079-00.pdf"),
        ("BAC-BH-0182-00 GalaxySAR & Climate Controller", "ZVP-PC-0079-00.pdf"),
        ("BAC-BH-0183-00 Galaxy SAR With Cape & Breathing Tube", "ZVP-PC-0079-00.pdf"),
    ],
    # Categories with no documentation (empty placeholder on PanBlast site):
    "closed_circuit_blast_equipment": [],
    "operator_protective_equipment": [],
    "pressure_and_suction_blast_cabinets": [],
    "pressure_blast_machines": [],
    "respirator_air_line_filters_and_accessories": [],
    "specialty_tools_and_equipment": [],
}


def download(url: str, dest: Path, *, retries: int = 3, timeout: int = 60) -> int:
    """Download `url` to `dest`. Returns the number of bytes written."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*;q=0.8",
    }
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return len(data)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_err = exc
            wait = 2 * attempt
            print(f"  attempt {attempt}/{retries} failed: {exc}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url}: {last_err}")


def unique_pdfs(manuals: dict[str, list[tuple[str, str]]]) -> Iterable[tuple[str, str]]:
    """Yield (category, pdf_filename) for every unique pdf in each category."""
    for category, entries in manuals.items():
        seen: set[str] = set()
        for _product, pdf in entries:
            if pdf in seen:
                continue
            seen.add(pdf)
            yield category, pdf


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"
    data_root.mkdir(exist_ok=True)

    print(f"Downloading PanBlast manuals into: {data_root}")

    total_files = 0
    total_bytes = 0
    failures: list[str] = []

    for category, pdf in unique_pdfs(MANUALS):
        url = BASE_PDF_URL + pdf
        dest = data_root / category / pdf
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {category}/{pdf} (already exists, {dest.stat().st_size} bytes)")
            total_files += 1
            total_bytes += dest.stat().st_size
            continue
        print(f"[get ] {url} -> {dest.relative_to(project_root)}")
        try:
            size = download(url, dest)
            print(f"       wrote {size} bytes")
            total_files += 1
            total_bytes += size
        except Exception as exc:  # noqa: BLE001 - record and continue
            print(f"       ERROR: {exc}")
            failures.append(f"{category}/{pdf}: {exc}")

    manifest = {
        "source_url": "https://www.panblast.com/manuals.acv",
        "categories": {
            category: [
                {"product": product, "pdf": pdf, "path": f"{category}/{pdf}"}
                for product, pdf in entries
            ]
            for category, entries in MANUALS.items()
        },
    }
    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote manifest: {manifest_path.relative_to(project_root)}")

    print(
        f"\nDone. {total_files} file(s), {total_bytes / 1024:.1f} KB total. "
        f"{len(failures)} failure(s)."
    )
    if failures:
        print("\nFailures:")
        for line in failures:
            print(f"  - {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
