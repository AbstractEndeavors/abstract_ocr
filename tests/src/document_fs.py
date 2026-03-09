from pathlib import Path
import shutil


def normalize_pdf_directory(pdf_path):
##
    pdf = Path(pdf_path)

    if pdf.is_dir():
        return pdf

    dirname = pdf.parent
    name = pdf.stem

    target_dir = dirname / name

    if not target_dir.exists():
        target_dir.mkdir()

    new_pdf = target_dir / pdf.name

    if pdf != new_pdf:
        shutil.move(str(pdf), str(new_pdf))

    return target_dir
