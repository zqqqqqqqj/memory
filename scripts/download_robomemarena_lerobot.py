#!/usr/bin/env python3
"""Download and verify the user-selected LeRobot v3 dataset at a pinned revision."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = 'yinchenghust/robomemarena_lerobot'
REVISION = 'b6595d7af67749d3644056bab0e096194332c1ba'
DEST = ROOT / 'data/lerobot' / REPO


def session():
    s = requests.Session()
    s.trust_env = False
    return s


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def hashes(path):
    sha = hashlib.sha256()
    git = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
            git.update(chunk)
    return sha.hexdigest(), git.hexdigest()


def verify(path, item):
    if not path.exists() or path.stat().st_size != item['size']:
        return None
    sha, git = hashes(path)
    good = sha == item['lfs']['oid'] if 'lfs' in item else git == item['oid']
    return sha if good else None


def fetch(item, destination, endpoint):
    target = destination / item['path']
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError('Invalid repository path')
    target.parent.mkdir(parents=True, exist_ok=True)
    if digest := verify(target, item):
        return dict(path=item['path'], size=item['size'], sha256=digest)
    partial = target.with_name(target.name + '.partial')
    s = session()
    for attempt in range(6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > item['size']:
                partial.unlink()
                offset = 0
            if offset < item['size'] or not partial.exists():
                url = f'{endpoint}/datasets/{REPO}/resolve/{REVISION}/{quote(item["path"], safe="/")}'
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                with s.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        offset = 0
                    if offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected resume range')
                    with partial.open('ab' if offset else 'wb') as stream:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            stream.write(chunk)
            digest = verify(partial, item)
            if digest is None:
                if partial.exists() and partial.stat().st_size == item['size']:
                    partial.unlink()
                raise ValueError('Size or source checksum mismatch')
            partial.replace(target)
            print(f'VERIFIED {item["path"]} {item["size"]:,} bytes', flush=True)
            return dict(path=item['path'], size=item['size'], sha256=digest)
        except (requests.RequestException, ValueError) as error:
            print(f'RETRY {attempt+1} {item["path"]}: {error}', flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / 'download_source.json'
    if source.exists():
        document = json.loads(source.read_text())
        if document['revision'] != REVISION or document['repo_id'] != REPO:
            raise ValueError('Destination contains a different dataset revision')
    else:
        s = session()
        url = f'{args.endpoint}/api/datasets/{REPO}/tree/{REVISION}?recursive=true&expand=false&limit=1000'
        entries = []
        while url:
            response = s.get(url, timeout=(20, 60))
            response.raise_for_status()
            entries.extend(x for x in response.json() if x['type'] == 'file')
            url = response.links.get('next', {}).get('url')
        if not entries:
            raise RuntimeError('Empty dataset listing')
        document = {'repo_id': REPO, 'revision': REVISION, 'endpoint': args.endpoint,
                    'source_url': f'https://huggingface.co/datasets/{REPO}', 'files': entries}
        atomic_json(source, document)
    # Finder metadata is unrelated to dataset contents and the mirror blocks that dotfile.
    entries = sorted((x for x in document['files'] if Path(x['path']).name != '.DS_Store'),
                     key=lambda x: (not x['path'].startswith('meta/'), x['path']))
    status = {'repo_id': REPO, 'revision': REVISION, 'root': str(args.output),
              'total_files': len(entries), 'total_bytes': sum(x['size'] for x in entries),
              'excluded_non_dataset_files': [x['path'] for x in document['files'] if Path(x['path']).name == '.DS_Store'],
              'verified_files': [], 'errors': [], 'complete': False}
    status_path = args.output / 'download_manifest.json'
    atomic_json(status_path, status)
    print(f'Downloading {len(entries)} files, {status["total_bytes"]:,} bytes to {args.output}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(fetch, item, args.output, args.endpoint): item for item in entries}
        for future in as_completed(jobs):
            try:
                status['verified_files'].append(future.result())
            except Exception as error:
                status['errors'].append({'path': jobs[future]['path'], 'error': str(error)})
            status['updated_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(status_path, status)
    status['complete'] = len(status['verified_files']) == len(entries) and not status['errors']
    atomic_json(status_path, status)
    if not status['complete']:
        raise RuntimeError(f'Download incomplete; see {status_path}')
    print('COMPLETE: all repository files match upstream LFS SHA256 or Git blob SHA1.', flush=True)


if __name__ == '__main__':
    main()
