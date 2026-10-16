"""Verify the seven display figures against their recorded SHA-256 values."""
from pathlib import Path
import hashlib,json,struct
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'asset_manifest.json').read_text())
assert len(rows)==7
for row in rows:
    data=(ROOT/row['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==row['sha256'],row['path']
    assert data[:8]==b'\x89PNG\r\n\x1a\n'
    assert struct.unpack('>II',data[16:24])==(row['width'],row['height'])
print('PASS: seven display figure hashes and dimensions.')
