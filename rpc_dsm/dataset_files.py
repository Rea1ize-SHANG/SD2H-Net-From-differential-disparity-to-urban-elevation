"""Write self-contained dataset files without symbolic or hard links."""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_real_parents(path):
    for parent in Path(path).absolute().parents:
        if parent.is_symlink():
            raise ValueError('Destination parent is a link; materialize it first: ' + str(parent))


def copy_dataset_file(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if not source.is_file():
        raise ValueError('Expected a regular source file: ' + str(source))
    require_real_parents(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_symlink():
        if source == destination:
            return False
        if source.stat().st_size != destination.stat().st_size or file_hash(source) != file_hash(destination):
            raise ValueError('Existing destination differs; refusing to overwrite: ' + str(destination))
        return False
    size = source.stat().st_size
    if shutil.disk_usage(destination.parent).free < size + 64 * 1024 ** 2:
        raise OSError('Insufficient space for an atomic file copy: ' + str(destination))
    descriptor, temporary = tempfile.mkstemp(prefix='.' + destination.name + '.',
                                            suffix='.copying', dir=str(destination.parent))
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        before = source.stat()
        shutil.copyfile(source, temporary)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Source changed while copying: ' + str(source))
        if file_hash(source) != file_hash(temporary):
            raise RuntimeError('Copied file checksum mismatch: ' + str(destination))
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def source_files(root, ancestors=None):
    root = Path(root).resolve(strict=True)
    ancestors = set() if ancestors is None else ancestors
    if root in ancestors:
        raise ValueError('Directory link cycle: ' + str(root))
    for child in sorted(root.iterdir()):
        source = child.resolve(strict=True)
        if source.is_dir():
            for relative, path in source_files(source, ancestors | {root}):
                yield Path(child.name) / relative, path
        elif source.is_file():
            yield Path(child.name), source
        else:
            raise ValueError('Unsupported dataset entry: ' + str(child))


def copy_dataset_tree(source, destination, on_file=None, workers=1):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    require_real_parents(destination)
    entries = list(source_files(source))
    temporary = backup = None
    target = destination
    if destination.is_symlink():
        needed = sum(path.stat().st_size for _, path in entries)
        if shutil.disk_usage(destination.parent).free < needed + 64 * 1024 ** 2:
            raise OSError('Insufficient space to materialize directory: ' + str(destination))
        temporary = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.', dir=str(destination.parent)))
        target = temporary
    target.mkdir(parents=True, exist_ok=True)
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            pending = {pool.submit(copy_dataset_file, path, target / relative): path
                       for relative, path in entries}
            for future in as_completed(pending):
                future.result()
                if on_file:
                    on_file(pending[future].stat().st_size)
        if temporary is not None:
            backup = destination.with_name('.' + destination.name + '.link-backup')
            if backup.exists() or backup.is_symlink():
                raise ValueError('Recovery backup already exists: ' + str(backup))
            os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
            backup.unlink()
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
