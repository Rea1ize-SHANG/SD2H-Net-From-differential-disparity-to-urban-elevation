import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from osgeo import ogr, osr
from rasterio.transform import from_origin

from rpc_dsm.building_heights import estimate_instance_heights, process_scene as summarize_scene_heights
from rpc_dsm.inputs import discover, find_pair, scene_id
from rpc_dsm.project import warp_scene
from rpc_dsm.rectify import write_global_blocks
from rpc_dsm.crop import process_scene
from rpc_dsm.raster_labels import build_scene as build_raster_scene
from rpc_dsm.watch import snapshot
from rpc_dsm import triangulate
from rpc_dsm.vector_labels import build_scene as build_vector_scene


class WorkflowTests(unittest.TestCase):
    def test_stereo_discovery_excludes_multispectral(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'GF07_DLC_016431_20221014_KR111_01_058_L1A_01'
            folder.mkdir()
            for band in ('BPA', 'FPA', 'BMS'):
                (folder / ('GF07_' + band + '_016431_20221014_KR111_01_058_L1A_01.tif')).touch()
            left, right = find_pair(folder)
            self.assertIn('_BPA_', left.name)
            self.assertIn('_FPA_', right.name)
            self.assertEqual(scene_id(folder), '058')
            self.assertEqual(list(discover(Path(temporary))), ['058'])

    def test_raster_label_is_geographically_warped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, city = '058', 'GF7_058'
            for folder in (root / 'tiles' / 'left' / city, root / 'crop' / scene,
                           root / 'geo' / city):
                folder.mkdir(parents=True)
            (root / 'tiles' / 'left' / city / 'GF7_058_0_0_left.tif').touch()
            (root / 'crop' / scene / 'common_crop.json').write_text(json.dumps({'window': [0, 0, 1024, 1024]}))
            coords = np.arange(65, dtype=np.float32) * 16
            longitude = 110 + coords[np.newaxis, :] * 0.00001 + np.zeros((65, 1), dtype=np.float32)
            latitude = 25 - coords[:, np.newaxis] * 0.00001 + np.zeros((1, 65), dtype=np.float32)
            with rasterio.open(root / 'geo' / city / 'GF7_058_lonlat_stride16.tif', 'w',
                               driver='GTiff', width=65, height=65, count=2, dtype='float32') as out:
                out.write(longitude, 1)
                out.write(latitude, 2)
            source = np.zeros((1024, 1024), dtype=np.float32)
            source[200:240, 300:340] = 12
            label_path = root / 'height.tif'
            with rasterio.open(label_path, 'w', driver='GTiff', width=1024, height=1024,
                               count=1, dtype='float32', crs='EPSG:4326',
                               transform=from_origin(110, 25, 0.00001, 0.00001)) as out:
                out.write(source, 1)
            with rasterio.open(label_path) as raster:
                self.assertEqual(build_raster_scene(scene, raster, root / 'tiles', root / 'labels',
                                             root / 'crop', root / 'geo'), 1)
            with rasterio.open(root / 'labels' / 'label' / city / 'GF7_058_0_0_label.tif') as image:
                self.assertEqual(image.read(1)[220, 320], 12)

    def test_iqr_excludes_roof_outlier(self):
        surface = np.full((100, 100), 2.0, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[40:60, 40:60] = 73
        surface[mask > 0] = 12.0
        surface[45, 45] = 1000.0
        truth = np.full((100, 100), 10.0, dtype=np.float32)
        measurements = estimate_instance_heights(surface, mask, truth, 1.0, 20.0)
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0][:3], (73, 10.0, 10.0))

    def test_rpc_geogrid_produces_projected_dsm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dsm, grid, output = (root / name for name in ('dsm.tif', 'geo.tif', 'projected.tif'))
            with rasterio.open(dsm, 'w', driver='GTiff', width=64, height=64,
                               count=1, dtype='float32', nodata=-9999) as image:
                image.write(np.full((64, 64), 12.5, np.float32), 1)
            with rasterio.open(grid, 'w', driver='GTiff', width=4, height=4,
                               count=2, dtype='float32') as image:
                rows, columns = np.indices((4, 4))
                image.write((110.0 + columns * 16e-5).astype(np.float32), 1)
                image.write((25.0 - rows * 16e-5).astype(np.float32), 2)
            warp_scene(dsm, grid, output, stride=16, resolution=2)
            with rasterio.open(output) as image:
                self.assertEqual(image.crs.to_epsg(), 32649)
                self.assertTrue(np.any(image.read(1) == 12.5))

    def test_vector_label_without_original_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, city = '058', 'GF7_058'
            for folder in (root / 'tiles' / 'left' / city, root / 'tiles' / 'right' / city,
                           root / 'crop' / scene, root / 'geo' / city):
                folder.mkdir(parents=True)
            (root / 'tiles' / 'left' / city / 'GF7_058_0_0_left.tif').touch()
            (root / 'tiles' / 'right' / city / 'GF7_058_0_0_right.tif').touch()
            (root / 'crop' / scene / 'common_crop.json').write_text(json.dumps({'window': [0, 0, 1024, 1024]}))
            coords = np.arange(65, dtype=np.float32) * 16
            longitude = 110 + coords[np.newaxis, :] * 0.00001 + np.zeros((65, 1), dtype=np.float32)
            latitude = 25 - coords[:, np.newaxis] * 0.00001 + np.zeros((1, 65), dtype=np.float32)
            with rasterio.open(root / 'geo' / city / 'GF7_058_lonlat_stride16.tif', 'w',
                               driver='GTiff', width=65, height=65, count=2, dtype='float32') as out:
                out.write(longitude, 1)
                out.write(latitude, 2)
            shape_path = root / 'buildings.shp'
            source = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(str(shape_path))
            spatial = osr.SpatialReference()
            spatial.ImportFromEPSG(4326)
            layer = source.CreateLayer('buildings', spatial, ogr.wkbPolygon)
            layer.CreateField(ogr.FieldDefn('Height', ogr.OFTReal))
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for lon, lat in ((110.003, 24.997), (110.0035, 24.997),
                             (110.0035, 24.9965), (110.003, 24.9965), (110.003, 24.997)):
                ring.AddPoint(lon, lat)
            polygon = ogr.Geometry(ogr.wkbPolygon)
            polygon.AddGeometry(ring)
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(polygon)
            feature.SetField('Height', 12.0)
            layer.CreateFeature(feature)
            feature = None
            source = None
            count, _ = build_vector_scene(scene, root / 'tiles', root / 'labels', shape_path,
                                   root / 'progress.json', 1, 1, stable_mask=True,
                                   crop_root=root / 'crop', geo_root=root / 'geo', city_prefix='GF7_')
            self.assertEqual(count, 1)
            with rasterio.open(root / 'labels' / 'mask' / city / 'GF7_058_0_0_mask.tif') as mask:
                self.assertEqual(mask.read(1)[320, 320], 1)

    def test_monitor_exits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'progress.json').write_text('{"stage":"DONE"}')
            self.assertEqual(snapshot(path), ('DONE', True))

    def test_rpc_intersection_uses_numeric_crop_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / '058'
            folder.mkdir()
            for name in ('global_H_ref_crop.txt', 'global_H_sec_crop.txt'):
                np.savetxt(folder / name, np.eye(3))
            with patch.object(triangulate, 'CROP_ROOT', Path(temporary)), patch.object(
                    triangulate, 'solve_batch', return_value=(np.array([10.]),
                     np.array([0.]), np.array([True]))):
                height, residual = triangulate.solve_disparity_block(
                    None, 'GF7_058', np.zeros((1, 1), np.float32), 0, 0,
                    None, None, np.eye(3), 1, 3.0, lambda *_: None)
            self.assertEqual(float(height[0, 0]), 10.0)
            self.assertEqual(float(residual[0, 0]), 0.0)

    def test_whole_image_rectification_and_common_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'epipolar' / '058'
            folder.mkdir(parents=True)
            raw = root / 'raw.tif'
            values = np.full((1024, 1024), 7.0, dtype=np.float32)
            with rasterio.open(raw, 'w', driver='GTiff', width=1024, height=1024,
                               count=1, dtype='float32') as image:
                image.write(values, 1)
            for side in ('ref', 'sec'):
                write_global_blocks(folder / ('rectified_' + side + '_global.tif'),
                                    raw, np.eye(3), 1024, 1024)
                np.savetxt(folder / ('global_H_' + side + '.txt'), np.eye(3))
            info = process_scene(root, '058', '058')
            self.assertTrue(info['verified'])
            self.assertEqual(info['window'], [0, 0, 1024, 1024])
            for side in ('ref', 'sec'):
                with rasterio.open(root / 'crop' / '058' / ('rectified_' + side + '_common.tif')) as image:
                    self.assertEqual(float(image.read(1)[512, 512]), 7.0)

    def test_aligned_labels_produce_building_height_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            city = 'GF7_058'
            for folder in (root / 'dsm' / city, root / 'geo' / city,
                           root / 'labels' / 'mask' / city, root / 'labels' / 'label' / city):
                folder.mkdir(parents=True)
            building = np.zeros((1024, 1024), np.int32)
            building[500:520, 500:520] = 73
            surface = np.full((1024, 1024), 2.0, np.float32)
            surface[building > 0] = 12.0
            for path, pixels in ((root / 'dsm' / city / (city + '_DSM.tif'), surface),
                                 (root / 'labels' / 'mask' / city / (city + '_0_0_mask.tif'), building),
                                 (root / 'labels' / 'label' / city / (city + '_0_0_label.tif'),
                                  np.where(building > 0, 10, 0).astype(np.float32))):
                with rasterio.open(path, 'w', driver='GTiff', width=1024, height=1024,
                                   count=1, dtype=pixels.dtype) as image:
                    image.write(pixels, 1)
            coords = np.arange(65, dtype=np.float32) * 16
            with rasterio.open(root / 'geo' / city / (city + '_lonlat_stride16.tif'), 'w',
                               driver='GTiff', width=65, height=65, count=2, dtype='float32') as image:
                image.write(110 + coords[np.newaxis, :] * 0.00001 + np.zeros((65, 1), np.float32), 1)
                image.write(25 - coords[:, np.newaxis] * 0.00001 + np.zeros((1, 65), np.float32), 2)
            args = argparse.Namespace(dsm_root=root / 'dsm', geo_root=root / 'geo',
                                      labels_root=root / 'labels', output_root=root / 'heights',
                                      buffer_m=20.0)
            self.assertEqual(summarize_scene_heights('058', args), 1)
            with (root / 'heights' / (city + '.csv')).open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]['building_id'], '73')
            self.assertEqual(float(rows[0]['pred_height_m']), 10.0)


if __name__ == '__main__':
    unittest.main()
