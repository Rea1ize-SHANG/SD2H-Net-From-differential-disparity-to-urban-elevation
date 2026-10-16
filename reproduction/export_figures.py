"""Export web figures from matching manuscript PDFs without editing the PDFs."""
from pathlib import Path
import argparse, hashlib, json, subprocess, tempfile
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--figures', type=Path, required=True, help='Directory of original manuscript figure PDFs')
parser.add_argument('--output', type=Path, required=True, help='New directory for exported PNG files')
args = parser.parse_args()
assert not args.output.exists(), 'Choose a new output directory to preserve accepted display assets.'
manifest = json.loads((ROOT/'asset_manifest.json').read_text())
for row in manifest:
    source = args.figures/row['source_asset']
    assert source.is_file(), str(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_sha256'], str(source)
args.output.mkdir(parents=True)
# Keep temporary renders inside the explicitly chosen writable output directory.
with tempfile.TemporaryDirectory(dir=args.output) as temporary:
    for row in manifest:
        rendered = Path(temporary)/row['name']
        subprocess.run(['pdftoppm','-f','1','-singlefile','-scale-to',str(row['render_max_dimension']),
                        '-png',str(args.figures/row['source_asset']),str(rendered)], check=True)
        image = Image.open(rendered.with_suffix('.png')).convert('RGB')
        assert list(image.size) == row['source_render_dimensions']
        if row['panel_crop_pixels']:
            image = image.crop(tuple(row['panel_crop_pixels']))
        difference = ImageChops.difference(image, Image.new('RGB',image.size,'white')).convert('L')
        bounds = difference.point(lambda v:255 if v>18 else 0).getbbox()
        if bounds:
            bounds = (max(0,bounds[0]-16),max(0,bounds[1]-16),min(image.width,bounds[2]+16),min(image.height,bounds[3]+16))
            image = image.crop(bounds)
        assert image.size == (row['width'],row['height'])
        target = args.output/Path(row['path']).name
        image.save(target,optimize=True)
        print(target.name)
print('Export completed. Source PDFs were not modified.')
