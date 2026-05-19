# 🎯 YOLOv5 실시간 객체 탐지 웹앱

> EmmettHwang/yolov5ReadTime 프로젝트를 스마트폰에서 바로 실행 가능한 웹앱으로 구현

## 📋 프로젝트 개요
<img width="1280" height="1032" alt="image" src="https://github.com/user-attachments/assets/a81fac6f-1cd2-4791-84a8-3cde4b5dcd06" />


- **원본 프로젝트**: [yolov5ReadTime](https://github.com/EmmettHwang/yolov5ReadTime)
- **구현 방식**: TensorFlow.js + COCO-SSD (MobileNet V2) - 브라우저 내 AI 추론
- **지원 클래스**: COCO 80개 객체 (YOLOv5와 동일)
- **플랫폼**: 모바일/데스크톱 브라우저 (Chrome, Safari, Edge)
- **PWA 지원**: 홈 화면 설치 가능

## ✅ 구현된 기능

- **실시간 객체 탐지** - 스마트폰/웹캠 카메라 스트림에서 실시간 탐지
- **바운딩 박스 시각화** - 컬러 박스, 코너 마커, 라벨 오버레이
- **신뢰도 표시** - 각 객체별 퍼센트 신뢰도
- **FPS / 추론 시간 측정** - 실시간 성능 통계
- **프레임 스킵** - 1~5 조절로 성능 최적화 (원본 기능 동일)
- **신뢰도 임계값 조절** - 10~95% 슬라이더
- **최대 탐지 수 조절** - 1~50개
- **카메라 전환** - 전면/후면 카메라 즉시 전환
- **이미지 캡처 저장** - 탐지 결과 포함 JPEG 저장
- **탐지 결과 목록** - 클래스별 신뢰도 리스트 (중복 합산)
- **PWA** - 홈 화면 추가로 앱처럼 사용
- **오프라인 캐싱** - Service Worker로 재방문 시 빠르게 로드

## 📁 프로젝트 구조

```
/
├── index.html          # 메인 HTML (SPA)
├── manifest.json       # PWA 매니페스트
├── sw.js               # Service Worker
├── css/
│   └── style.css       # 전체 스타일 (다크 테마)
├── js/
│   └── app.js          # 메인 로직 (TensorFlow.js 탐지)
└── icons/
    ├── icon.svg        # SVG 아이콘 소스
    └── icon-512.png    # PNG 아이콘 (PWA용)
```

## 🚀 진입점

| 경로 | 설명 |
|------|------|
| `index.html` | 메인 앱 (유일한 페이지) |
| `manifest.json` | PWA 설정 |
| `sw.js` | 오프라인 캐시 |

## 🤖 AI 모델

| 항목 | 원본 (Python) | 웹앱 (Browser) |
|------|--------------|----------------|
| 모델 | YOLOv5s (PyTorch) | COCO-SSD MobileNetV2 (TF.js) |
| 클래스 수 | COCO 80개 | COCO 80개 (동일) |
| 추론 위치 | 로컬 Python | 브라우저 (클라이언트) |
| 서버 불필요 | ❌ (Python 환경 필요) | ✅ (브라우저만으로 동작) |

## ⚙️ 설정 옵션

| 설정 | 범위 | 기본값 |
|------|------|--------|
| 신뢰도 임계값 | 10~95% | 50% |
| 프레임 스킵 | 1~5 | 1 |
| 최대 탐지 수 | 1~50 | 20 |
| 바운딩 박스 | on/off | on |
| 라벨 표시 | on/off | on |
| 신뢰도 표시 | on/off | on |
| 카메라 | 전면/후면 | 후면 |

## 📱 스마트폰 사용 방법

1. Chrome/Safari에서 웹앱 URL 접속
2. 카메라 권한 허용
3. **"탐지 시작"** 버튼 탭
4. 카메라를 객체에 향하면 실시간 탐지 시작
5. (선택) 홈 화면에 추가 → 앱처럼 사용

## 🔧 미구현 / 향후 개선

- [ ] YOLOv5 네이티브 모델 직접 변환 (ONNX → TF.js)
- [ ] 동영상 파일 업로드 후 탐지
- [ ] 탐지 이력 저장 (Table API 활용)
- [ ] 커스텀 클래스 필터 (특정 객체만 탐지)
- [ ] 탐지 통계 차트 (시간별)
- [ ] 아이콘 PNG 파일 실제 생성

## 📚 참고

- [TensorFlow.js](https://www.tensorflow.org/js)
- [COCO-SSD 모델](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd)
- [YOLOv5 원본](https://github.com/ultralytics/yolov5)

---
## 실행 
https://emmetthwang.github.io/yolowebapp/
**작성일**: 2026-05-19
