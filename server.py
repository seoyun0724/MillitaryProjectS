"""Local annotation and inference application. Python standard library only."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
PHOTOS = ROOT.parent / '차량 사진'
DATA = ROOT / 'data'
CLASSES = ['bumper', 'hood', 'door', 'fender', 'headlight', 'taillight', 'mirror', 'wheel', 'windshield', 'grille', 'scratch', 'dent', 'crack', 'corrosion', 'broken']
LABELS = ['범퍼', '보닛', '문', '펜더', '전조등', '후미등', '사이드미러', '바퀴', '유리', '그릴', '긁힘', '찌그러짐', '균열', '부식', '파손']
VEHICLES = {'rexton_sports': '렉스턴 스포츠', 'korando': '코란도'}

def vehicle_key(value):
    if value not in VEHICLES:
        raise ValueError('차종을 선택하세요.')
    return value

def photos():
    return sorted(p for p in PHOTOS.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}) if PHOTOS.exists() else []

def validate_boxes(boxes):
    if not isinstance(boxes, list):
        raise ValueError('상자 목록이 필요합니다.')
    for box in boxes:
        if type(box.get('class_id')) is not int or not 0 <= box['class_id'] < len(CLASSES):
            raise ValueError('잘못된 분류입니다.')
        coords = [box.get(k) for k in ('x', 'y', 'w', 'h')]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in coords):
            raise ValueError('좌표는 유한한 숫자여야 합니다.')
        x, y, w, h = coords
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > 1.000001 or y+h > 1.000001:
            raise ValueError('상자가 이미지 범위를 벗어났습니다.')
    return boxes

def annotation_path(photo, vehicle='rexton_sports'):
    return DATA / vehicle_key(vehicle) / 'annotations' / (photo.name + '.json')

def export_dataset(vehicle='rexton_sports'):
    vehicle_key(vehicle)
    import shutil
    reviewed = []
    for p in photos():
        a = annotation_path(p, vehicle)
        if a.exists():
            reviewed.append((p, validate_boxes(json.loads(a.read_text('utf-8'))['boxes'])))
    if len(reviewed) < 2:
        raise ValueError('최소 2장의 사진을 검토·저장해야 분할할 수 있습니다.')
    # Keep exports immutable so older labels cannot remain in a new dataset.
    from datetime import datetime
    dest = DATA / vehicle / ('dataset_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    reviewed.sort(key=lambda item: hashlib.sha256(item[0].read_bytes()).hexdigest())
    nval = max(1, round(len(reviewed) * .2))
    for i, (p, boxes) in enumerate(reviewed):
        split = 'val' if i < nval else 'train'
        for kind in ('images', 'labels'):
            (dest / kind / split).mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(p.read_bytes()).hexdigest()[:16] + '_' + p.name
        shutil.copy2(p, dest / 'images' / split / name)
        lines = ['{} {:.8f} {:.8f} {:.8f} {:.8f}'.format(b['class_id'], b['x']+b['w']/2, b['y']+b['h']/2, b['w'], b['h']) for b in boxes]
        (dest / 'labels' / split / Path(name).with_suffix('.txt')).write_text('\n'.join(lines), 'utf-8')
    config = 'path: ' + json.dumps(dest.as_posix(), ensure_ascii=False) + '\ntrain: images/train\nval: images/val\nnames:\n'
    config += ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES))
    (dest / 'data.yaml').write_text(config, 'utf-8')
    return {'path': str(dest / 'data.yaml'), 'train': len(reviewed)-nval, 'val': nval}

class Handler(BaseHTTPRequestHandler):
    def reply(self, value, status=200, content_type='application/json; charset=utf-8'):
        data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def photo(self, index):
        items = photos()
        index = int(index)
        if index < 0 or index >= len(items):
            raise ValueError('사진을 찾을 수 없습니다.')
        return items[index]

    def do_GET(self):
        try:
            url = urlparse(self.path)
            vehicle = vehicle_key(parse_qs(url.query).get('vehicle', ['rexton_sports'])[0])
            if url.path == '/':
                return self.reply((ROOT / 'index.html').read_bytes(), content_type='text/html; charset=utf-8')
            if url.path == '/api/photos':
                return self.reply({'photos': [{'id': i, 'name': p.name, 'reviewed': annotation_path(p, vehicle).exists()} for i, p in enumerate(photos())], 'classes': CLASSES, 'labels': LABELS, 'model_ready': (ROOT / 'models' / vehicle / 'best.pt').exists()})
            p = self.photo(parse_qs(url.query).get('id', ['-1'])[0])
            if url.path == '/api/image':
                import mimetypes
                return self.reply(p.read_bytes(), content_type=mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
            if url.path == '/api/annotation':
                a = annotation_path(p, vehicle)
                return self.reply(json.loads(a.read_text('utf-8')) if a.exists() else {'boxes': [], 'reviewed': False})
            self.reply({'error': 'Not found'}, 404)
        except (ValueError, OSError) as e:
            self.reply({'error': str(e)}, 400)

    def do_POST(self):
        try:
            # Only the local application's own origin may submit changes.
            if self.headers.get('Origin') not in (None, f'http://{self.headers.get("Host")}'):
                return self.reply({'error': 'Origin rejected'}, 403)
            size = int(self.headers.get('Content-Length', 0))
            if size < 0 or size > 2_000_000:
                raise ValueError('요청 크기 초과')
            body = json.loads(self.rfile.read(size) or '{}')
            vehicle = vehicle_key(body.get('vehicle', 'rexton_sports'))
            if self.path == '/api/export':
                return self.reply(export_dataset(vehicle))
            p = self.photo(body['id'])
            if self.path == '/api/annotation':
                boxes = validate_boxes(body['boxes'])
                other = next(v for v in VEHICLES if v != vehicle)
                if annotation_path(p, other).exists():
                    raise ValueError('이 사진은 다른 차종으로 이미 저장되어 있습니다. 사진과 차종을 확인하세요.')
                a = annotation_path(p, vehicle)
                a.parent.mkdir(parents=True, exist_ok=True)
                temp = a.with_suffix('.tmp')
                temp.write_text(json.dumps({'boxes': boxes, 'reviewed': True}, ensure_ascii=False), 'utf-8')
                temp.replace(a)
                return self.reply({'saved': True})
            if self.path == '/api/predict':
                weights = ROOT / 'models' / vehicle / 'best.pt'
                if not weights.exists():
                    raise ValueError(f'미학습: models/{vehicle}/best.pt에 해당 차종 전용 모델을 넣어 주세요.')
                result = subprocess.run([sys.executable, str(ROOT / 'model.py'), 'predict', '--image', str(p), '--vehicle', vehicle, '--weights', str(weights)], capture_output=True, text=True, encoding='utf-8', timeout=180)
                if result.returncode:
                    raise ValueError(result.stderr[-2000:] or '모델 실행 실패')
                return self.reply(json.loads(result.stdout))
            self.reply({'error': 'Not found'}, 404)
        except (ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired) as e:
            self.reply({'error': str(e)}, 400)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    print(f'차량 외관 검사: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
