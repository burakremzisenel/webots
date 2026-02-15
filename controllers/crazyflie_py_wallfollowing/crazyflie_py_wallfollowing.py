# -*- coding: utf-8 -*-
#
#  ...........       ____  _ __
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/

# MIT License

# Copyright (c) 2023 Bitcraze


"""
file: crazyflie_py_wallfollowing.py

Controls the crazyflie and implements a wall following method in webots in Python

Author:   Kimberly McGuire (Bitcraze AB)
"""


from controller import Robot
from controller import Keyboard

# Importe fuer Bildverarbeitung
import numpy as np
import cv2
from ultralytics import YOLO
import collections  # Moving Average için

from math import cos, sin, sqrt, atan2

# =====================================================
# KALMAN FİLTER - Pozisyon ve Hız Tahmini için
# OcSort/DeepOcSort implementasyonundan adapte edildi
# =====================================================
class SimpleKalmanFilter:
    """
    2D Kalman Filter for tracking target position and velocity.
    State: [x, y, vx, vy] - position and velocity
    Measurement: [x, y] - only position is observed
    """
    def __init__(self, initial_x=0, initial_y=0):
        # State: [x, y, vx, vy]
        self.x = np.array([[initial_x], [initial_y], [0.0], [0.0]])

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1],  # vy = vy
        ], dtype=float)

        # Measurement matrix (only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Initial covariance (high uncertainty)
        self.P = np.eye(4) * 100

        # Process noise
        self.Q = np.diag([1.0, 1.0, 0.5, 0.5])

        # Measurement noise
        self.R = np.diag([5.0, 5.0])

        # Track state
        self.hits = 0
        self.time_since_update = 0
        self.age = 0

    def predict(self):
        """Predict next state based on constant velocity model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.x[:2].flatten()  # Return predicted [x, y]

    def update(self, z):
        """Update state with measurement z = [x, y]."""
        z = np.array([[z[0]], [z[1]]])

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        self.hits += 1
        self.time_since_update = 0

        return self.x[:2].flatten()  # Return updated [x, y]

    def get_velocity(self):
        """Get current velocity estimate [vx, vy]."""
        return self.x[2:].flatten()

    def get_position(self):
        """Get current position estimate [x, y]."""
        return self.x[:2].flatten()

    def get_speed(self):
        """Get speed magnitude."""
        vx, vy = self.get_velocity()
        return sqrt(vx**2 + vy**2)

    def get_direction(self):
        """Get velocity direction in radians."""
        vx, vy = self.get_velocity()
        return atan2(vy, vx)


# =====================================================
# TRACK STATE - Hedef durumu yönetimi
# =====================================================
class TrackState:
    INITIALIZING = 0  # İlk birkaç frame (henüz onaylanmamış)
    CONFIRMED = 1     # Onaylanmış hedef
    LOST = 2          # Hedef kaybedildi (aranıyor)


def velocity_direction_consistency(current_pos, predicted_pos, last_pos):
    """
    Check if the detection is consistent with predicted velocity direction.
    Returns a score between 0 and 1 (1 = perfectly consistent).
    """
    if last_pos is None:
        return 1.0

    # Predicted direction
    pred_dx = predicted_pos[0] - last_pos[0]
    pred_dy = predicted_pos[1] - last_pos[1]
    pred_norm = sqrt(pred_dx**2 + pred_dy**2) + 1e-6

    # Actual direction
    act_dx = current_pos[0] - last_pos[0]
    act_dy = current_pos[1] - last_pos[1]
    act_norm = sqrt(act_dx**2 + act_dy**2) + 1e-6

    # Cosine similarity
    cos_sim = (pred_dx * act_dx + pred_dy * act_dy) / (pred_norm * act_norm)

    # Convert to score (0-1)
    score = (cos_sim + 1) / 2
    return score

from pid_controller import pid_velocity_fixed_height_controller
from wall_following import WallFollowing

FLYING_ATTITUDE = 1

def detect_object(image_data, width, height, model, mode='group'):
    """
    YENİ MANTIK:
    mode='group': Tüm nesneleri grupla, büyük box'a dön (renk filtresi YOK)
    mode='select': Renk filtresi uygula ve bir hedef seç
    """
    if image_data is None:
        return None

    # Görüntüyü dönüştür
    img_array = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
    img_bgr = np.array(img_array[:, :, :3])

    # YOLO Algılama
    results = model(img_bgr, verbose=False, stream=True)

    all_objects = []  # Tüm nesneler

    for result in results:
        for box in result.boxes:
            # Araba, şişe, bardak, vazo (COCO dataset classes)
            # 2=car, 39=bottle, 41=cup, 75=vase
            if int(box.cls) in [2, 39, 41, 75]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = float(box.xywh[0][0])
                cy = float(box.xywh[0][1])
                all_objects.append({'x': cx, 'y': cy, 'box': (x1, y1, x2, y2)})

    if mode == 'group':
        # GRUP MODU: Büyük bounding box hesapla
        if len(all_objects) == 0:
            cv2.imshow("YOLO Sichtfeld", img_bgr)
            cv2.waitKey(1)
            return None

        # Tüm box'ların min/max'ını bul
        all_x1 = min(obj['box'][0] for obj in all_objects)
        all_y1 = min(obj['box'][1] for obj in all_objects)
        all_x2 = max(obj['box'][2] for obj in all_objects)
        all_y2 = max(obj['box'][3] for obj in all_objects)

        # Grup merkezini hesapla
        group_cx = (all_x1 + all_x2) / 2
        group_cy = (all_y1 + all_y2) / 2

        # Tüm nesneleri sarı çiz
        for obj in all_objects:
            x1, y1, x2, y2 = obj['box']
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # Büyük grup box'ı MAVİ çiz
        cv2.rectangle(img_bgr, (all_x1, all_y1), (all_x2, all_y2), (255, 0, 0), 3)
        cv2.putText(img_bgr, "GRUP", (all_x1, all_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("YOLO Sichtfeld", img_bgr)
        cv2.waitKey(1)

        return {'x': group_cx, 'y': group_cy, 'box': (all_x1, all_y1, all_x2, all_y2), 'count': len(all_objects)}

    elif mode == 'select':
        # SEÇİM MODU: Renk filtresi uygula
        red_candidates = []

        for obj in all_objects:
            x1, y1, x2, y2 = obj['box']

            # ROI çıkar ve renk kontrolü yap
            roi = img_bgr[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]

            is_red = False
            if roi.size > 0:
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv_roi, np.array([0, 50, 20]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv_roi, np.array([170, 50, 20]), np.array([180, 255, 255]))
                red_ratio = cv2.countNonZero(mask1 + mask2) / (roi.shape[0] * roi.shape[1])

                if red_ratio > 0.1:
                    is_red = True

            if is_red:
                red_candidates.append(obj)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Sarı
            else:
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (100, 100, 100), 1)  # Gri (kırmızı değil)

        # Ekran merkezine en yakın kırmızıyı seç
        target = None
        if red_candidates:
            center_x = width / 2
            red_candidates.sort(key=lambda c: abs(c['x'] - center_x))
            target = red_candidates[0]

            # Hedefi YEŞİL çiz
            tx1, ty1, tx2, ty2 = target['box']
            cv2.rectangle(img_bgr, (tx1, ty1), (tx2, ty2), (0, 255, 0), 3)
            cv2.putText(img_bgr, "HEDEF", (tx1, ty1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO Sichtfeld", img_bgr)
        cv2.waitKey(1)

        if target:
            return target
        return None

    return None

if __name__ == '__main__':

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # ===== DRONE TİPİ ALGILAMA =====
    # Crazyflie mi yoksa Mavic 2 Pro mu kullanılıyor?
    drone_type = "UNKNOWN"

    # Crazyflie kontrolü (m1_motor varsa Crazyflie)
    test_motor = robot.getDevice("m1_motor")
    if test_motor is not None:
        drone_type = "CRAZYFLIE"
        print("=== DRONE TİPİ: CRAZYFLIE ===")

        # Crazyflie Motorları
        m1_motor = robot.getDevice("m1_motor")
        m1_motor.setPosition(float('inf'))
        m1_motor.setVelocity(-1)
        m2_motor = robot.getDevice("m2_motor")
        m2_motor.setPosition(float('inf'))
        m2_motor.setVelocity(1)
        m3_motor = robot.getDevice("m3_motor")
        m3_motor.setPosition(float('inf'))
        m3_motor.setVelocity(-1)
        m4_motor = robot.getDevice("m4_motor")
        m4_motor.setPosition(float('inf'))
        m4_motor.setVelocity(1)
        motors = [m1_motor, m2_motor, m3_motor, m4_motor]

        # Crazyflie Sensörleri
        imu = robot.getDevice("inertial_unit")
        gps = robot.getDevice("gps")
        gyro = robot.getDevice("gyro")

        # Crazyflie Range Sensörleri
        range_front = robot.getDevice("range_front")
        range_left = robot.getDevice("range_left")
        range_back = robot.getDevice("range_back")
        range_right = robot.getDevice("range_right")
        has_range_sensors = True

    else:
        # Mavic 2 Pro kontrolü
        test_motor = robot.getDevice("front left propeller")
        if test_motor is not None:
            drone_type = "MAVIC2PRO"
            print("=== DRONE TİPİ: MAVIC 2 PRO ===")

            # Mavic 2 Pro Motorları
            m1_motor = robot.getDevice("front left propeller")
            m1_motor.setPosition(float('inf'))
            m1_motor.setVelocity(1)
            m2_motor = robot.getDevice("front right propeller")
            m2_motor.setPosition(float('inf'))
            m2_motor.setVelocity(-1)
            m3_motor = robot.getDevice("rear left propeller")
            m3_motor.setPosition(float('inf'))
            m3_motor.setVelocity(-1)
            m4_motor = robot.getDevice("rear right propeller")
            m4_motor.setPosition(float('inf'))
            m4_motor.setVelocity(1)
            motors = [m1_motor, m2_motor, m3_motor, m4_motor]

            # Mavic 2 Pro Sensörleri (BÜYÜK HARF!)
            imu = robot.getDevice("inertial unit")
            gps = robot.getDevice("gps")
            gyro = robot.getDevice("gyro")

            # Mavic 2 Pro Range Sensörü (cameraSlot'a eklendiyse)
            range_front = robot.getDevice("range_front")
            range_left = None
            range_back = None
            range_right = None
            if range_front is not None:
                has_range_sensors = True
                print("[BİLGİ] Mavic 2 Pro'da range_front sensörü bulundu!")
            else:
                has_range_sensors = False
                print("[UYARI] Mavic 2 Pro'da range sensörü yok! Box-tabanlı mesafe kullanılacak.")

        else:
            print("=== HATA: Bilinmeyen drone tipi! ===")
            exit(1)

    # Sensörleri etkinleştir
    imu.enable(timestep)
    gps.enable(timestep)
    gyro.enable(timestep)

    # Kamera (her iki drone için aynı)
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Range sensörlerini etkinleştir (varsa)
    if has_range_sensors:
        if range_front is not None:
            range_front.enable(timestep)
        if range_left is not None:
            range_left.enable(timestep)
        if range_back is not None:
            range_back.enable(timestep)
        if range_right is not None:
            range_right.enable(timestep)

    # Klavye
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Değişkenleri başlat
    past_x_global = 0
    past_y_global = 0
    past_time = robot.getTime()
    first_time = True

    # YOLO Model yükle (yolov8n.pt en küçük/hızlı model)
    # İlk çalıştırmada otomatik indirilir
    yolo_model = YOLO('yolov8n.pt')

    # Crazyflie hız PID kontrolcüsü
    PID_crazyflie = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    height_desired = FLYING_ATTITUDE

    wall_following = WallFollowing(angle_value_buffer=0.01, reference_distance_from_wall=0.5,
                                   max_forward_speed=0.3, init_state=WallFollowing.StateWallFollowing.FORWARD)

    autonomous_mode = False

    # Görev Durumu: YENİ MANTIK
    mission_state = "TAKEOFF"  # TAKEOFF -> SEARCH_GROUP -> ALIGN_GROUP -> SELECT_TARGET -> APPROACH
    mission_timer = 0.0
    locked_target = None  # Seçilen hedef
    group_aligned = False  # Grup merkezine hizalandık mı?
    max_box_height = 0  # Maksimum box yüksekliği (schrumpfeni tespit için)

    # ===== KALMAN FILTER & TRACK STATE (BoxMOT/OcSort'tan) =====
    target_kalman = None  # Hedef için Kalman filter
    track_state = TrackState.INITIALIZING  # Track durumu
    MIN_HITS_TO_CONFIRM = 3  # Onay için gereken ardışık algılama
    MAX_AGE_LOST = 15  # Kaybedilmeden önce maksimum frame sayısı
    last_target_pos = None  # Son hedef pozisyonu (velocity direction için)

    # DEAD ZONE - Küçük hataları yoksay (jitter'ı önler)
    DEAD_ZONE_X = 15  # piksel - X ekseninde dead zone
    DEAD_ZONE_Y = 10  # piksel - Y ekseninde dead zone

    # VELOCITY DIRECTION CONSISTENCY
    MIN_VELOCITY_CONSISTENCY = 0.3  # Minimum tutarlılık skoru (0-1)

    # MOVING AVERAGE FİLTRESİ - Target jumping'i önlemek için
    MA_X_LEN = 5  # X pozisyonu için 5 örnek
    MA_Z_LEN = 5  # Mesafe/BoxHeight için 5 örnek
    MA_DIST_LEN = 5  # Distanz sensörü için 5 örnek
    ma_x = collections.deque(maxlen=MA_X_LEN)  # X pozisyonu Moving Average
    ma_box_height = collections.deque(maxlen=MA_Z_LEN)  # Box yüksekliği Moving Average
    ma_distance = collections.deque(maxlen=MA_DIST_LEN)  # Distanz sensörü Moving Average

    print("\n")

    print("====== Controls =======\n\n")

    print(" The Crazyflie can be controlled from your keyboard!\n")
    print(" All controllable movement is in body coordinates\n")
    print("- Use the up, back, right and left button to move in the horizontal plane\n")
    print("- Use Q and E to rotate around yaw\n ")
    print("- Use W and S to go up and down\n ")
    print("- Press A to start autonomous mode\n")
    print("- Press D to disable autonomous mode\n")

    # Ana döngü:
    while robot.step(timestep) != -1:

        dt = robot.getTime() - past_time
        mission_timer += dt

        # Simülasyon adımı çok küçükse veya başlangıçta sıfıra bölmeyi önle
        if dt == 0:
            continue

        actual_state = {}

        if first_time:
            past_x_global = gps.getValues()[0]
            past_y_global = gps.getValues()[1]
            past_time = robot.getTime()
            first_time = False

        # Sensör verilerini al
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global)/dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global)/dt
        altitude = gps.getValues()[2]

        # Gövde sabit hızları al
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = - v_x_global * sin_yaw + v_y_global * cos_yaw

        # Değerleri başlat
        desired_state = [0, 0, 0, 0]
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0

        key = keyboard.getKey()
        while key > 0:
            if key == Keyboard.UP:
                forward_desired += 0.5
            elif key == Keyboard.DOWN:
                forward_desired -= 0.5
            elif key == Keyboard.RIGHT:
                sideways_desired -= 0.5
            elif key == Keyboard.LEFT:
                sideways_desired += 0.5
            elif key == ord('Q'):
                yaw_desired = + 1
            elif key == ord('E'):
                yaw_desired = - 1
            elif key == ord('W'):
                height_diff_desired = 0.1
            elif key == ord('S'):
                height_diff_desired = - 0.1
            elif key == ord('A'):
                if autonomous_mode is False:
                    autonomous_mode = True
                    print("Autonomous mode: ON")
            elif key == ord('D'):
                if autonomous_mode is True:
                    autonomous_mode = False
                    print("Autonomous mode: OFF")
            key = keyboard.getKey()

        height_desired += height_diff_desired * dt

        # --- YENİ GRUP BAZLI GÖREV MANTIĞI ---
        camera_data = camera.getImage()

        # Range sensör değerlerini oku (varsa)
        if has_range_sensors and range_front is not None:
            range_front_value = range_front.getValue() / 1000
        else:
            range_front_value = 2.0  # Max range (uzak varsay)

        if has_range_sensors and range_right is not None:
            range_right_value = range_right.getValue() / 1000
        else:
            range_right_value = 2.0

        if has_range_sensors and range_left is not None:
            range_left_value = range_left.getValue() / 1000
        else:
            range_left_value = 2.0

        if mission_state == "TAKEOFF":
            # 2 saniye yüksel
            if mission_timer > 2.0:
                print("=== KALKIŞ TAMAMLANDI - GRUP ARAMAYA BAŞLA ===")
                mission_state = "SEARCH_GROUP"
                mission_timer = 0

        elif mission_state == "SEARCH_GROUP":
            # GRUP MODU: Tüm nesneleri grupla (renk filtresi YOK)
            group_info = detect_object(camera_data, camera.getWidth(), camera.getHeight(), yolo_model, mode='group')

            # Yavaşça dön
            yaw_desired = 0.3

            if group_info is not None:
                print(f"=== GRUP BULUNDU! {group_info['count']} nesne, Box={group_info['box']} - HİZALAMAYA BAŞLA ===")
                mission_state = "ALIGN_GROUP"
                mission_timer = 0
            elif mission_timer > 20.0:
                print("=== GRUP ARAMA ZAMAN AŞIMI ===")
                mission_timer = 0

        elif mission_state == "ALIGN_GROUP":
            # GRUP ORTALAMA HİZALAMA
            group_info = detect_object(camera_data, camera.getWidth(), camera.getHeight(), yolo_model, mode='group')

            if group_info is not None:
                group_cx = group_info['x']
                center_x = camera.getWidth() / 2
                error_x = group_cx - center_x

                # Grubun merkezine dön
                yaw_desired = -0.003 * error_x
                yaw_desired = max(-0.4, min(0.4, yaw_desired))

                print(f"[HİZALAMA] Grup merkezi X={group_cx:.0f}, Hata={error_x:.0f}, Yaw={yaw_desired:.2f}")

                # Hizalandık mı? (±50 piksel tolerans)
                if abs(error_x) < 50:
                    if not group_aligned:
                        group_aligned = True
                        mission_timer = 0  # Stabilizasyon için zamanlayıcı sıfırla

                    # 1 saniye stabil kalırsa hedef seçimine geç
                    if mission_timer > 1.0:
                        print("=== GRUP HİZALANDI - HEDEF SEÇİMİNE GEÇ ===")
                        mission_state = "SELECT_TARGET"
                        mission_timer = 0
                        group_aligned = False
                else:
                    group_aligned = False
            else:
                # Grup kaybedildi
                print("=== GRUP KAYBEDİLDİ - ARAMAYA GERİ DÖN ===")
                mission_state = "SEARCH_GROUP"
                mission_timer = 0
                group_aligned = False

        elif mission_state == "SELECT_TARGET":
            # HEDEF SEÇİMİ: Renk filtresi uygula
            target_info = detect_object(camera_data, camera.getWidth(), camera.getHeight(), yolo_model, mode='select')

            if target_info is not None:
                locked_target = {'x': target_info['x'], 'y': target_info['y'], 'box': target_info['box']}
                print(f"=== KIRMIZI HEDEF SEÇİLDİ X={target_info['x']:.0f}, Box={target_info['box']}, Mesafe={range_front_value:.2f}m - YAKLAŞMAYA BAŞLA ===")
                mission_state = "APPROACH"
                mission_timer = 0

                # ===== KALMAN FILTER BAŞLAT =====
                target_kalman = SimpleKalmanFilter(initial_x=target_info['x'], initial_y=target_info['y'])
                track_state = TrackState.INITIALIZING
                last_target_pos = (target_info['x'], target_info['y'])

                # Moving Average'ı sıfırla ve ilk değeri ekle
                ma_x.clear()
                ma_box_height.clear()
                ma_distance.clear()
                ma_x.append(target_info['x'])
                box_h = target_info['box'][3] - target_info['box'][1]
                ma_box_height.append(box_h)
                max_box_height = box_h
            else:
                # Kırmızı nesne yok, gruba geri dön
                if mission_timer > 3.0:
                    print("=== KIRMIZI HEDEF BULUNAMADI - GRUP ARAMAYA GERİ DÖN ===")
                    mission_state = "SEARCH_GROUP"
                    mission_timer = 0

        elif mission_state == "APPROACH":
            # YAKLAŞMA: Kilitli hedefe uç (KALMAN + VELOCITY CONSISTENCY + DEAD ZONE)
            if locked_target is not None and target_kalman is not None:
                # ===== KALMAN PREDICT (önce tahmin yap) =====
                predicted_pos = target_kalman.predict()

                # Yeni algılama yap
                current_target = detect_object(camera_data, camera.getWidth(), camera.getHeight(), yolo_model, mode='select')

                detection_valid = False
                if current_target is not None:
                    current_pos = (current_target['x'], current_target['y'])

                    # ===== VELOCITY DIRECTION CONSISTENCY =====
                    # Algılanan pozisyon, tahmin edilen yönle tutarlı mı?
                    vdc_score = velocity_direction_consistency(current_pos, predicted_pos, last_target_pos)

                    # Kalman hızı yeterince büyükse VDC kontrolü yap
                    if target_kalman.get_speed() > 2.0:
                        if vdc_score < MIN_VELOCITY_CONSISTENCY:
                            # Tutarsız algılama - muhtemelen yanlış hedef
                            print(f"[UYARI] VDC düşük: {vdc_score:.2f} - algılama reddedildi")
                            detection_valid = False
                        else:
                            detection_valid = True
                    else:
                        # Düşük hızda VDC kontrolü gevşet
                        detection_valid = True

                    if detection_valid:
                        # ===== KALMAN UPDATE =====
                        filtered_pos = target_kalman.update(current_pos)
                        last_target_pos = current_pos

                        # Moving Average'a da ekle
                        ma_x.append(filtered_pos[0])
                        new_box = current_target['box']
                        new_box_height = new_box[3] - new_box[1]
                        ma_box_height.append(new_box_height)

                        # Box'ı güncelle
                        locked_target['box'] = new_box
                        locked_target['y'] = filtered_pos[1]
                        locked_target['x'] = filtered_pos[0]

                        # Track state güncelle
                        if track_state == TrackState.INITIALIZING:
                            if target_kalman.hits >= MIN_HITS_TO_CONFIRM:
                                track_state = TrackState.CONFIRMED
                                print(f"=== HEDEF ONAYLANDI (hits={target_kalman.hits}) ===")

                # Algılama yoksa veya geçersizse, Kalman tahmini kullan
                if not detection_valid:
                    # Tahmini pozisyonu kullan
                    locked_target['x'] = predicted_pos[0]
                    locked_target['y'] = predicted_pos[1]

                    # Track durumunu kontrol et
                    if target_kalman.time_since_update > MAX_AGE_LOST:
                        track_state = TrackState.LOST
                        print(f"=== HEDEF KAYBEDİLDİ (time_since_update={target_kalman.time_since_update}) ===")
                        mission_state = "SEARCH_GROUP"
                        mission_timer = 0
                        target_kalman = None
                        locked_target = None
                        last_target_pos = None
                        ma_x.clear()
                        ma_box_height.clear()
                        ma_distance.clear()
                        max_box_height = 0
                        continue  # Döngüye devam et

                # MOVING AVERAGE hesapla (Kalman + MA kombinasyonu)
                if len(ma_x) > 0:
                    smoothed_x = sum(ma_x) / len(ma_x)
                else:
                    smoothed_x = locked_target['x']

                if len(ma_box_height) > 0:
                    smoothed_box_height = sum(ma_box_height) / len(ma_box_height)
                else:
                    smoothed_box_height = 20

                object_x = smoothed_x  # Kalman + MA filtrelenmiş
                object_y = locked_target['y']
                target_box = locked_target['box']

                center_x = camera.getWidth() / 2
                center_y = camera.getHeight() / 2
                error_x = object_x - center_x
                error_y = object_y - center_y

                # ===== DEAD ZONE - Küçük hataları yoksay =====
                if abs(error_x) < DEAD_ZONE_X:
                    error_x = 0
                if abs(error_y) < DEAD_ZONE_Y:
                    error_y = 0

                # ===== LIDAR_ON_TARGET CHECK =====
                box_x1, box_y1, box_x2, box_y2 = target_box
                lidar_on_target = (box_x1 < center_x < box_x2) and (box_y1 < center_y < box_y2)

                # Eğer range sensörü varsa VE lidar hedefe bakıyorsa
                if has_range_sensors and lidar_on_target and range_front_value < 2.0 and range_front_value > 0.1:
                    ma_distance.append(range_front_value)

                # Distanz için YUMUŞATILMIŞ değer
                if len(ma_distance) > 0:
                    smoothed_distance = sum(ma_distance) / len(ma_distance)
                else:
                    smoothed_distance = 2.0

                # ===== YAW KONTROL (Dead zone uygulanmış) =====
                if error_x != 0:
                    yaw_desired = -0.004 * error_x
                    yaw_desired = max(-0.5, min(0.5, yaw_desired))
                else:
                    yaw_desired = 0  # Dead zone içinde - dönme

                # Box yüksekliği - YUMUŞATILMIŞ
                box_height = smoothed_box_height

                # Maksimum box yüksekliğini takip et
                if box_height > max_box_height:
                    max_box_height = box_height

                # STOPP LOGIC: Box schrumpft
                is_shrinking = (max_box_height - box_height) > 10

                # PHASE DETECTION
                is_close_phase = box_height >= 35

                # --- YÜKSEK SEVİYE KONTROL (Z ekseni) ---
                if is_close_phase and error_y != 0:
                    height_diff_desired = -error_y * 0.0002
                    height_diff_desired = max(-0.08, min(0.08, height_diff_desired))

                # --- İLERİ HAREKET (sadece CONFIRMED track için) ---
                if track_state == TrackState.CONFIRMED and abs(error_x) < 80:
                    if is_close_phase and lidar_on_target and len(ma_distance) >= 3:
                        if smoothed_distance > 0.35:
                            forward_desired = 0.10
                        elif smoothed_distance > 0.25:
                            forward_desired = 0.05
                        else:
                            forward_desired = 0
                            print(f"=== HEDEFE ULAŞILDI! MA_Dist={smoothed_distance:.2f}m ===")
                    else:
                        if is_shrinking:
                            forward_desired = 0
                            print(f"=== HEDEFE ULAŞILDI! BoxH={box_height:.0f}px ===")
                        else:
                            forward_desired = 0.15 if not is_close_phase else 0.10
                elif track_state == TrackState.INITIALIZING:
                    # Onay beklerken yavaş ilerle
                    forward_desired = 0.05

                # Her 10. frame'de log
                if int(mission_timer * 32) % 10 == 0:
                    phase = "YAKIN" if is_close_phase else "UZAK"
                    if has_range_sensors:
                        lidar_status = "ON" if lidar_on_target else "OFF"
                        dist_info = f"Dist={smoothed_distance:.2f}m"
                    else:
                        lidar_status = "NO"
                        dist_info = "BoxOnly"

                    # Track state string
                    state_str = ["INIT", "CONF", "LOST"][track_state]
                    kf_speed = target_kalman.get_speed() if target_kalman else 0

                    print(f"[{state_str}] {phase}, KF_X={object_x:.0f}, BoxH={box_height:.0f}px, V={kf_speed:.1f}, LiDAR={lidar_status}, {dist_info}, Fwd={forward_desired:.2f}")
            else:
                # Hedef kaybedildi
                print("=== HEDEF KAYBEDİLDİ - GRUP ARAMAYA GERİ DÖN ===")
                mission_state = "SEARCH_GROUP"
                mission_timer = 0
                max_box_height = 0
                ma_x.clear()
                ma_box_height.clear()
                ma_distance.clear()

        # Duvar takibi yönü seç
        # Sol yön seçersen, sağ mesafe değerini kullan
        # Sağ yön seçersen, sol mesafe değerini kullan
        direction = WallFollowing.WallFollowingDirection.LEFT
        range_side_value = range_right_value

        # Duvar takibi durum makinesinden hız komutlarını al
        cmd_vel_x, cmd_vel_y, cmd_ang_w, state_wf = wall_following.wall_follower(
            range_front_value, range_side_value, yaw, direction, robot.getTime())

        if autonomous_mode:
            sideways_desired = cmd_vel_y
            forward_desired = cmd_vel_x
            yaw_desired = cmd_ang_w

        # Motor hızlarını ayarla (drone tipine göre farklı kontrol)
        if drone_type == "CRAZYFLIE":
            # Crazyflie PID kontrolcüsü
            motor_power = PID_crazyflie.pid(dt, forward_desired, sideways_desired,
                                            yaw_desired, height_desired,
                                            roll, pitch, yaw_rate,
                                            altitude, v_x, v_y)
            m1_motor.setVelocity(-motor_power[0])
            m2_motor.setVelocity(motor_power[1])
            m3_motor.setVelocity(-motor_power[2])
            m4_motor.setVelocity(motor_power[3])

        elif drone_type == "MAVIC2PRO":
            # ===== MAVIC 2 PRO ÖZEL KONTROL SİSTEMİ =====
            # Webots örnek kontrolcüsünden alındı (mavic2pro.c)

            # Sabitler (ampirik olarak bulunmuş)
            K_VERTICAL_THRUST = 68.5  # Bu thrust ile drone kalkar
            K_VERTICAL_OFFSET = 0.6   # Hedef irtifa ofseti
            K_VERTICAL_P = 3.0        # Dikey PID P sabiti
            K_ROLL_P = 50.0           # Roll PID P sabiti
            K_PITCH_P = 30.0          # Pitch PID P sabiti

            # Gyro değerlerini al (roll/pitch velocity)
            roll_velocity = gyro.getValues()[0]
            pitch_velocity = gyro.getValues()[1]

            # Disturbance değerlerini hesapla (desired değerlerden)
            # forward_desired -> pitch_disturbance (negatif = ileri)
            # sideways_desired -> roll_disturbance (pozitif = sol)
            # yaw_desired -> yaw_disturbance
            # Webots örneğinde keyboard ile -2.0 kullanılıyor
            # forward_desired=0.15 için -2.0 civarı çıkması lazım -> 0.15 * 13 = ~2
            pitch_disturbance = -forward_desired * 13.0  # Ölçek faktörü artırıldı
            roll_disturbance = sideways_desired * 2.0
            yaw_disturbance = yaw_desired * 1.3

            # Clamp fonksiyonu
            def clamp(value, low, high):
                return max(low, min(high, value))

            # Girişleri hesapla
            roll_input = K_ROLL_P * clamp(roll, -1, 1) + roll_velocity + roll_disturbance
            pitch_input = K_PITCH_P * clamp(pitch, -1, 1) + pitch_velocity + pitch_disturbance
            yaw_input = yaw_disturbance

            # Dikey kontrol (kübik fonksiyon)
            clamped_diff_alt = clamp(height_desired - altitude + K_VERTICAL_OFFSET, -1, 1)
            vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3)

            # Motor karışımı (Webots Mavic 2 Pro formülü)
            front_left = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
            front_right = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
            rear_left = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
            rear_right = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input

            # Motor hızlarını ayarla (işaretler Webots örneğinden)
            m1_motor.setVelocity(front_left)       # front left: pozitif
            m2_motor.setVelocity(-front_right)     # front right: negatif
            m3_motor.setVelocity(-rear_left)       # rear left: negatif
            m4_motor.setVelocity(rear_right)       # rear right: pozitif

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global
