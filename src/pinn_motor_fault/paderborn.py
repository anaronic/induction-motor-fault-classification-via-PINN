"""Download and load the Paderborn University Bearing Dataset."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import urllib.parse
import urllib.request

import numpy as np
from tqdm import tqdm

OFFICIAL_BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
BEARING_CODE_RE = re.compile(r"\b(K(?:00\d|A\d{2}|I\d{2}|B\d{2}))\b", re.IGNORECASE)


class DatasetError(RuntimeError):
    """Raised when dataset acquisition or parsing fails."""


@dataclass(frozen=True)
class DownloadedArchive:
    code: str
    path: Path
    url: str


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def infer_label_from_code(code: str) -> str:
    normalized = code.upper()
    if normalized.startswith("K00"):
        return "healthy"
    if normalized.startswith("KI"):
        return "inner_race"
    if normalized.startswith("KA"):
        return "outer_race"
    if normalized.startswith("KB"):
        return "rolling_element"
    raise DatasetError(f"Cannot infer class label from bearing code: {code}")


def infer_label_from_path(path: Path) -> str:
    match = BEARING_CODE_RE.search(path.name.upper())
    if not match:
        for part in path.parts:
            match = BEARING_CODE_RE.search(part.upper())
            if match:
                break
    if not match:
        raise DatasetError(f"Cannot infer bearing code from path: {path}")
    return infer_label_from_code(match.group(1))


def list_available_archives(base_url: str = OFFICIAL_BASE_URL) -> dict[str, str]:
    with urllib.request.urlopen(base_url, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    parser = _HrefParser()
    parser.feed(html)
    archives: dict[str, str] = {}
    for href in parser.hrefs:
        if not href.lower().endswith(".rar"):
            continue
        code = Path(urllib.parse.urlparse(href).path).stem.upper()
        archives[code] = urllib.parse.urljoin(base_url, href)
    if not archives:
        raise DatasetError(f"No .rar archives found at {base_url}")
    return archives


def download_archives(
    codes: list[str],
    raw_dir: Path,
    base_url: str = OFFICIAL_BASE_URL,
    overwrite: bool = False,
) -> list[DownloadedArchive]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    available = list_available_archives(base_url)
    downloaded: list[DownloadedArchive] = []
    for code in [item.upper() for item in codes]:
        if code not in available:
            raise DatasetError(f"{code} is not listed by the Paderborn archive index.")
        url = available[code]
        destination = raw_dir / f"{code}.rar"
        if destination.exists() and not overwrite:
            downloaded.append(DownloadedArchive(code=code, path=destination, url=url))
            continue
        print(f"Downloading {code} from {url}")
        with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
        downloaded.append(DownloadedArchive(code=code, path=destination, url=url))
    return downloaded


def extract_archives(archives: list[DownloadedArchive], extract_dir: Path) -> list[Path]:
    extractor = _find_extractor()
    if extractor is None:
        raise DatasetError(
            "RAR extraction needs one of these tools on PATH: 7z, 7za, unrar, or rar. "
            "The archives were downloaded successfully; install an extractor and rerun with --extract."
        )
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_dirs: list[Path] = []
    for archive in archives:
        target = extract_dir / archive.code
        target.mkdir(parents=True, exist_ok=True)
        extractor_name = Path(extractor).name.lower()
        if extractor_name in {"7z", "7za", "7z.exe", "7za.exe"}:
            command = [extractor, "x", "-y", f"-o{target}", str(archive.path)]
        elif extractor_name in {"tar", "tar.exe", "bsdtar", "bsdtar.exe"}:
            command = [extractor, "-xf", str(archive.path), "-C", str(target)]
        else:
            command = [extractor, "x", "-o+", str(archive.path), str(target)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise DatasetError(f"Failed to extract {archive.path}: {result.stderr or result.stdout}")
        extracted_dirs.append(target)
    return extracted_dirs


def _find_extractor() -> str | None:
    for name in ("tar", "bsdtar", "7z", "7za", "unrar", "rar"):
        found = shutil.which(name)
        if found:
            return found
    return None


def load_paderborn_windows(
    data_dir: Path,
    window_size: int = 4096,
    stride: int = 2048,
    signal_preference: str = "current",
    max_windows_per_file: int | None = 20,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mat_files = sorted(data_dir.rglob("*.mat"))
    if not mat_files:
        raise DatasetError(f"No .mat files found under {data_dir}")

    windows: list[np.ndarray] = []
    labels: list[str] = []
    sources: list[str] = []
    skipped: list[tuple[Path, str]] = []
    for mat_path in tqdm(mat_files, desc='Loading .mat files'):
        try:
            signal = load_signal_from_mat(mat_path, signal_preference=signal_preference)
            label = infer_label_from_path(mat_path)
        except Exception as exc:
            skipped.append((mat_path, f"{type(exc).__name__}: {exc}"))
            continue
        file_windows = make_windows(signal, window_size=window_size, stride=stride)
        if max_windows_per_file is not None:
            file_windows = file_windows[:max_windows_per_file]
        windows.extend(file_windows)
        labels.extend([label] * len(file_windows))
        sources.extend([str(mat_path)] * len(file_windows))

    if not windows:
        details = "; ".join(f"{path}: {reason}" for path, reason in skipped[:5])
        raise DatasetError(f"MAT files were found, but no signal windows could be created. Skipped examples: {details}")
    for path, reason in skipped:
        print(f"warning: skipped unreadable MAT file {path}: {reason}")
    return np.stack(windows).astype(np.float64), np.asarray(labels), sources


def load_signal_from_mat(path: Path, signal_preference: str = "current") -> np.ndarray:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise DatasetError("Install scipy to read Paderborn .mat files: python -m pip install -e .[paderborn]") from exc

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    candidates: list[tuple[int, np.ndarray, str]] = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        _collect_numeric_series(value, key, signal_preference, candidates)
    if not candidates:
        raise DatasetError(f"No usable numeric signal found in {path}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    signal = np.asarray(candidates[0][1], dtype=np.float64).reshape(-1)
    signal = signal[np.isfinite(signal)]
    if signal.size < 128:
        raise DatasetError(f"Signal in {path} is too short after parsing.")
    return signal


def _collect_numeric_series(
    value: object,
    path: str,
    signal_preference: str,
    candidates: list[tuple[int, np.ndarray, str]],
) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for field in value.dtype.names:
                _collect_numeric_series(value[field], f"{path}.{field}", signal_preference, candidates)
            return
        if value.dtype == object:
            for index, item in np.ndenumerate(value):
                _collect_numeric_series(item, f"{path}{index}", signal_preference, candidates)
            return
        if np.issubdtype(value.dtype, np.number) and value.size >= 512:
            score = _signal_score(path, signal_preference, value.size)
            candidates.append((score, np.asarray(value).reshape(-1), path))
            return
    if hasattr(value, "_fieldnames"):
        fields = getattr(value, "_fieldnames")
        if "Data" in fields and "Name" in fields:
            channel_name = _stringify_matlab_value(getattr(value, "Name"))
            channel_path = f"{path}.{channel_name}.Data" if channel_name else f"{path}.Data"
            _collect_numeric_series(getattr(value, "Data"), channel_path, signal_preference, candidates)
            return
        for field in fields:
            _collect_numeric_series(getattr(value, field), f"{path}.{field}", signal_preference, candidates)


def _stringify_matlab_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.dtype.kind in {"U", "S"}:
            return "".join(str(item) for item in value.reshape(-1)).strip()
        if value.dtype == object:
            return " ".join(_stringify_matlab_value(item) for item in value.reshape(-1)).strip()
    return str(value)


def _signal_score(path: str, preference: str, size: int) -> int:
    lowered = path.lower()
    score = min(size // 1024, 1000)
    if preference == "current":
        for token in ("current", "phase", "i_1", "i1", "i_2", "i2"):
            if token in lowered:
                score += 10_000
    elif preference == "vibration":
        for token in ("vibration", "acc", "bearing", "vib"):
            if token in lowered:
                score += 10_000
    if "data" in lowered:
        score += 1000
    if "time" in lowered or "speed" in lowered or "temp" in lowered:
        score -= 5000
    return score


def make_windows(signal: np.ndarray, window_size: int, stride: int) -> list[np.ndarray]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive.")
    if signal.size < window_size:
        return []
    return [signal[start : start + window_size].copy() for start in range(0, signal.size - window_size + 1, stride)]


def parse_shaft_frequency_hz(source: str | None, default_hz: float = 25.0) -> float:
    if not source:
        return default_hz
    match = re.search(r"N(\d{2})", source.upper())
    if not match:
        return default_hz
    return float(int(match.group(1)) * 100.0 / 60.0)
