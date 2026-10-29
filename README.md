# 차량 외관 인식 AI 개발 버전

현재 상태: 사진 검토·부품/손상 바운딩 박스 표시, YOLO 학습 데이터 내보내기, 전용 모델 추론 연결, JSON 검사 보고서 기능을 구현했습니다. **부품·손상 전용 학습 모델은 아직 없습니다.** 제공 사진 27장에 정답 데이터가 없으므로 자동 탐지 정확도를 주장할 수 없습니다.

## 실행

이 폴더에서 `python server.py`를 실행하고 브라우저에서 http://127.0.0.1:8765 를 여세요. Windows에서는 `start.cmd`를 더블 클릭해도 됩니다. 서버 종료는 터미널에서 Ctrl+C입니다. 화면·정답 데이터 구축에는 외부 패키지나 인터넷이 필요 없습니다. 상위 `차량 사진` 폴더의 사진을 읽으며 원본은 변경하지 않습니다.

1. 왼쪽에서 사진과 항목을 선택하고 드래그하여 위치를 표시합니다.
2. 사진에 보이는 모든 해당 부품과 손상을 표시한 뒤 검토 완료·저장합니다. 모호하거나 가려진 대상은 팀 기준을 정한 후 표시하세요. 손상은 부품 상자와 별도 상자로 표시합니다.
3. 저장한 2장 이상으로 학습 데이터를 내보냅니다. 실제 학습에는 다양한 차량·각도·조명 및 실제 손상 사례를 충분히 수집해야 합니다. 정상 사진도 모든 보이는 부품에 정답을 표시합니다.
4. `data/annotations`의 수동 정답과 내보낸 데이터는 로컬에 저장됩니다. 백업해 주세요. AI 결과는 수동 정답을 덮어쓰지 않습니다.

## 모델 학습과 평가

Python 3.11 또는 3.12의 별도 가상환경 사용을 권장합니다. 현재 시스템의 Python 3.14에서 AI 의존성 설치·실행은 검증하지 않았습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-ai.txt
.\.venv\Scripts\python model.py train --vehicle rexton_sports --data "내보낸 data.yaml 절대 경로" --epochs 100 --device cpu
```

처음 학습 시 기본 YOLO 가중치를 내려받습니다. GPU 환경은 호환되는 PyTorch 설치 후 `--device 0`을 사용하세요. 학습 종료 후 `runs/rexton_sports*/weights/best.pt`를 이 폴더의 `models/rexton_sports/best.pt`로 복사하고, 같은 가상환경 Python으로 `server.py`를 실행하세요. `models` 폴더는 직접 만들어 주세요. 프로그램은 클래스가 다른 일반 차량 탐지 모델을 거부합니다.

```powershell
.\.venv\Scripts\python model.py validate --vehicle rexton_sports --data "별도 평가 데이터의 data.yaml 절대 경로"
.\.venv\Scripts\python server.py
```

내보내기의 약 80:20 분할은 개발용입니다. 같은 차량·같은 촬영 세션·유사 이미지가 학습/평가 양쪽에 들어가면 성능이 부풀려집니다. 최종 성능은 차량 또는 촬영 세션 단위로 분리한 미사용 데이터에서 부품·손상별 precision/recall, mAP와 누락 사례를 확인하세요. 미검출을 정상으로 해석하지 않습니다. 부품 누락·변형·오염 등 아직 정의되지 않은 이상 상태 및 내부 고장은 현재 모델 범위에 포함하지 않습니다. 부품과 손상을 같은 화면에서 별도로 탐지하며, 손상의 해당 부품 자동 연결·심각도 판정은 아직 구현하지 않았습니다.

## 분류 기준

부품: bumper, hood, door, fender, headlight, taillight, mirror, wheel, windshield(차량 유리), grille.

손상: scratch, dent, crack, corrosion, broken.

실제 대상 차량을 확정한 뒤 범퍼 전/후 구분, 타이어/휠 구분, 정상/손상 경계, 가림 처리 기준 등을 팀에서 정해야 합니다. 분류를 바꾸면 기존 정답과 모델을 함께 변환해야 합니다.

## 검증 및 기술 참고

`python -m unittest discover -s . -p "test_*.py"`로 좌표 검증 및 YOLO 내보내기를 테스트합니다.

학습·추론 API: https://docs.ultralytics.com/usage/python/ . Ultralytics 사용·배포 전 해당 AGPL-3.0/상용 라이선스 적용 조건을 확인하세요. 현재 의존성 범위는 고정된 재현 환경이 아니므로 실제 학습 환경 확정 후 버전과 가중치 출처를 기록해야 합니다.

## 두 차종 별도 모델

화면에서 렉스턴 스포츠 또는 코란도를 먼저 선택하세요. 사진 목록은 공용이며 각 사진에 맞는 차종을 직접 선택합니다. 다른 차종으로 이미 저장한 사진의 중복 등록은 차단합니다. 정답과 내보내기 데이터는 data/rexton_sports와 data/korando로 분리합니다. 코란도 학습 명령은 --vehicle korando를 사용하며 결과를 models/korando/best.pt에 넣습니다. 각 차종에 대해 별도 데이터로 학습·평가를 실행해야 합니다. 현재 두 전용 모델 모두 미학습입니다. 차종 자동 식별은 구현 범위에 포함되지 않습니다.
