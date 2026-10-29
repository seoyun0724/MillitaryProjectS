"""Optional Ultralytics training/inference worker."""
import argparse
import contextlib
import json
import sys
from pathlib import Path
from server import ROOT, CLASSES, VEHICLES

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train', 'predict', 'validate'])
    p.add_argument('--data')
    p.add_argument('--image')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--vehicle', choices=list(VEHICLES), required=True)
    p.add_argument('--weights')
    p.add_argument('--base', default='yolo11n.pt')
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    a.weights = a.weights or str(ROOT / 'models' / a.vehicle / 'best.pt')
    with contextlib.redirect_stdout(sys.stderr):
        from ultralytics import YOLO
        if a.mode == 'train':
            if not a.data or not Path(a.data).is_file():
                p.error('--data에 내보낸 data.yaml 파일을 지정하세요.')
            model = YOLO(a.base)
            model.train(data=str(Path(a.data).resolve()), epochs=a.epochs, imgsz=960, batch=2, device=a.device, workers=0, project=str(ROOT / 'runs'), name=a.vehicle)
            print(f'학습 결과 best.pt를 models/{a.vehicle}/best.pt로 복사한 후 사용하세요.', file=sys.stderr)
            return
        if not Path(a.weights).is_file():
            raise ValueError('학습 모델 파일이 없습니다.')
        model = YOLO(a.weights)
        names = [model.names[i] for i in range(len(model.names))]
        if names != CLASSES:
            raise ValueError('모델 클래스가 이 프로젝트의 부품·손상 분류와 일치하지 않습니다.')
        if a.mode == 'validate':
            if not a.data:
                p.error('--data가 필요합니다.')
            model.val(data=a.data, device=a.device)
            return
        results = model.predict(source=a.image, conf=.25, imgsz=960, device=a.device, verbose=False)
        result = results[0]
        height, width = result.orig_shape
        boxes = []
        for b in result.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append({'class_id': int(b.cls.item()), 'confidence': float(b.conf.item()), 'x': x1/width, 'y': y1/height, 'w': (x2-x1)/width, 'h': (y2-y1)/height})
    print(json.dumps({'boxes': boxes, 'status': '검토 필요' if any(b['class_id'] >= 10 for b in boxes) else '손상 미검출 · 정상 여부 미확정'}, ensure_ascii=False))

if __name__ == '__main__':
    main()
