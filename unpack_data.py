"""Verify and safely extract the release archives without overwriting differences."""
from pathlib import Path
from zipfile import ZipFile
import hashlib,json
ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'metadata/archive_hashes.json').read_text())
    for name,digest in manifest.items():
        archive=ROOT/name
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('Archive hash mismatch: '+name)
        with ZipFile(archive) as z:
            for entry in z.infolist():
                target=(ROOT/entry.filename).resolve()
                if ROOT not in target.parents:raise ValueError('Unsafe archive path')
                data=z.read(entry)
                if target.exists():
                    if target.read_bytes()!=data:raise FileExistsError('Existing file differs: '+str(target))
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        print('Verified and extracted',name)

if __name__=='__main__':main()
