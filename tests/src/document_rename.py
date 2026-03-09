from pathlib import Path
import shutil
import re

PAGE_RE = re.compile(r"page[_\-]?(\d+)", re.IGNORECASE)


def rename_collection(directory, slug):

    directory = Path(directory)
    parent = directory.parent
    new_dir = parent / slug

    if new_dir.exists():
        raise RuntimeError("Target directory exists")

    for f in directory.rglob("*"):

        if not f.is_file():
            continue

        name = f.name

        if f.suffix == ".pdf":
            f.rename(f.with_name(f"{slug}.pdf"))

        else:
            m = PAGE_RE.search(name)
            if m:
                page = m.group(1)
                new_name = f"{slug}_page_{page}{f.suffix}"
                f.rename(f.with_name(new_name))

    shutil.move(str(directory), str(new_dir))

    return new_dir
