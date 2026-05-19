/**
 * Service Worker - YOLOv5 객체 탐지 PWA
 * 오프라인 지원 및 캐싱 전략
 */

const CACHE_NAME = 'yolov5-detect-v1';
const STATIC_ASSETS = [
  './',
  './index.html',
  './css/style.css',
  './js/app.js',
  './manifest.json',
];

// 설치 시 정적 파일 캐시
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(STATIC_ASSETS).catch(err => {
        console.warn('[SW] 일부 파일 캐싱 실패:', err);
      });
    })
  );
  self.skipWaiting();
});

// 오래된 캐시 정리
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

// 네트워크 우선, 실패 시 캐시 사용
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // TensorFlow.js CDN은 네트워크 우선 (캐시 저장 포함)
  if (url.hostname.includes('cdn.jsdelivr.net') || url.hostname.includes('tensorflow')) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async cache => {
        const cached = await cache.match(event.request);
        if (cached) return cached;
        try {
          const response = await fetch(event.request);
          if (response.ok) {
            cache.put(event.request, response.clone());
          }
          return response;
        } catch {
          return cached || new Response('Network error', { status: 503 });
        }
      })
    );
    return;
  }

  // 정적 파일: 캐시 우선
  if (event.request.method === 'GET') {
    event.respondWith(
      caches.match(event.request).then(cached => {
        if (cached) return cached;
        return fetch(event.request).then(response => {
          if (response.ok) {
            const toCache = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, toCache));
          }
          return response;
        }).catch(() => {
          return new Response('오프라인 상태입니다', { status: 503 });
        });
      })
    );
  }
});
