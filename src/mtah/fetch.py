"""Select the four tactical files out of the match bundle. Never the videos."""

import os
import shutil
import tarfile
import zipfile

WANTED = (
    "BAR-ATH_tactical_atd.txt",
    "BAR-ATH_tactical_atd.xml",
    "BAR-ATH_tactical_homography.csv",
    "BAR-ATH_tactical_smart_tagging.xml",
)


def _score(path):
    low = "/" + path.replace("\\", "/").lower().strip("/") + "/"
    tactical = 1 if "/tactical/" in low else 0
    return tactical, -len(path)


def choose_members(names):
    """Map each wanted basename to one archive member. Prefer a tactical path."""
    chosen = {}
    for name in names:
        if not name or name.endswith("/"):
            continue
        base = os.path.basename(name.rstrip("/"))
        if base not in WANTED:
            continue
        current = chosen.get(base)
        if current is None or _score(name) > _score(current):
            chosen[base] = name
    missing = [base for base in WANTED if base not in chosen]
    return chosen, missing


def _member_names(archive_path):
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            return archive.namelist()
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            return archive.getnames()
    raise IOError(
        "download is not a zip or tar archive. The secret URL must point at "
        "the bundle file itself."
    )


def extract_archive(archive_path, dest_dir):
    """Write the four tactical files into dest_dir. Return [(name, bytes)]."""
    chosen, missing = choose_members(_member_names(archive_path))
    if missing:
        raise IOError("archive is missing tactical files: {}".format(", ".join(missing)))

    os.makedirs(dest_dir, exist_ok=True)
    written = []

    def _save(stream, base):
        dest = os.path.join(dest_dir, base)
        with open(dest, "wb") as out:
            shutil.copyfileobj(stream, out)
        written.append((base, os.path.getsize(dest)))

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for base, member in chosen.items():
                with archive.open(member) as stream:
                    _save(stream, base)
    else:
        with tarfile.open(archive_path) as archive:
            for base, member in chosen.items():
                stream = archive.extractfile(member)
                if stream is None:
                    raise IOError("archive member is not a file: {}".format(base))
                with stream:
                    _save(stream, base)
    return written
