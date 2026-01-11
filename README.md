# Crazyflie Drone - YOLO Nesne Algılama Projesi

Webots simülasyonunda YOLO kullanarak kırmızı nesneleri algılayan ve onlara yaklaşan Crazyflie drone projesi.

## Özellikler

- ✅ YOLOv8 ile gerçek zamanlı nesne algılama
- ✅ Renk filtreleme (kırmızı nesneler)
- ✅ Hedef kilitleme ve takip
- ✅ Otonom kalkış, arama ve yaklaşma

## Kurulum

### Gereksinimler

- Webots R2023a veya üzeri
- Python 3.8+
- YOLO model (yolov8n.pt)

### Python Bağımlılıkları

```bash
pip install ultralytics opencv-python numpy
```

## Kullanım

1. Webots'ta world dosyasını aç
2. Simülasyonu başlat
3. Drone otomatik olarak:
   - Kalkış yapar
   - Dönüp kırmızı nesne arar
   - Bulduğunda yaklaşır

## Proje Yapısı

```
drone/
├── robots/
│   └── crazyflie/
│       └── controllers/
│           └── crazyflie_py_wallfollowing/
│               ├── crazyflie_py_wallfollowing.py
│               ├── pid_controller.py
│               └── wall_following.py
└── worlds/
    └── (world dosyaları)
```

## Lisans

MIT
