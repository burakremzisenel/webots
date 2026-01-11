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

from math import cos, sin

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
            # Sadece şişe, bardak, vazo
            if int(box.cls) in [39, 41, 75]:
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

    # Motorları başlat
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

    # Sensörleri başlat
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    range_front = robot.getDevice("range_front")
    range_front.enable(timestep)
    range_left = robot.getDevice("range_left")
    range_left.enable(timestep)
    range_back = robot.getDevice("range_back")
    range_back.enable(timestep)
    range_right = robot.getDevice("range_right")
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
        range_front_value = range_front.getValue() / 1000
        range_right_value = range_right.getValue() / 1000
        range_left_value = range_left.getValue() / 1000

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
            else:
                # Kırmızı nesne yok, gruba geri dön
                if mission_timer > 3.0:
                    print("=== KIRMIZI HEDEF BULUNAMADI - GRUP ARAMAYA GERİ DÖN ===")
                    mission_state = "SEARCH_GROUP"
                    mission_timer = 0

        elif mission_state == "APPROACH":
            # YAKLAŞMA: Kilitli hedefe uç
            if locked_target is not None:
                object_x = locked_target['x']
                object_y = locked_target['y']
                target_box = locked_target['box']

                center_x = camera.getWidth() / 2
                center_y = camera.getHeight() / 2
                error_x = object_x - center_x
                error_y = object_y - center_y  # Y ekseni hatası (yukarı/aşağı)

                # Hedefe dön (Yaw)
                yaw_desired = -0.004 * error_x
                yaw_desired = max(-0.5, min(0.5, yaw_desired))

                # Bounding box boyutunu kullanarak mesafe tahmin et
                box_height = target_box[3] - target_box[1]  # y2 - y1
                box_width = target_box[2] - target_box[0]   # x2 - x1

                # Maksimum box yüksekliğini takip et
                if box_height > max_box_height:
                    max_box_height = box_height

                # STOPP LOGIC: Box schrumpft (zumindest 10px kleiner als Maximum)
                is_shrinking = (max_box_height - box_height) > 10

                # PHASE DETECTION: Box büyüklüğüne göre faz belirle
                # Phase 1: Box < 35px → Uzak, sadece yatay yaklaş
                # Phase 2: Box >= 35px → Yakın, Z eksenini de ayarla + sensör kullan
                is_close_phase = box_height >= 35

                # --- YÜKSEK SEVİYE KONTROL (Z ekseni) ---
                if is_close_phase:
                    # YAKIN FAZ: Target'i Y ekseninde merkezle (yüksekliği ayarla)
                    # Hedef ekranın alt yarısındaysa → alçal
                    # Hedef ekranın üst yarısındaysa → yüksel
                    if abs(error_y) > 20:  # ±20px tolerans
                        height_diff_desired = -error_y * 0.0002  # Y hatası → yükseklik değişimi
                        height_diff_desired = max(-0.08, min(0.08, height_diff_desired))  # Sınırla

                # --- İLERİ HAREKET (X ekseni) ---
                if abs(error_x) < 80:  # X ekseninde merkezlenmişse
                    if is_close_phase:
                        # YAKIN FAZ: Distanz sensörü kullan
                        if range_front_value < 2.0 and range_front_value > 0.1:  # Geçerli okuma
                            if range_front_value > 0.35:
                                forward_desired = 0.10  # Yavaş yaklaş
                            elif range_front_value > 0.25:
                                forward_desired = 0.05  # Çok yavaş
                            else:
                                # 25cm'ye ulaştık → DUR
                                forward_desired = 0
                                print(f"=== HEDEFE ULAŞILDI! Distanz={range_front_value:.2f}m ===")
                        else:
                            # Sensör okuması yok, box boyutunu kullan
                            if is_shrinking:
                                forward_desired = 0
                                print(f"=== HEDEFE ULAŞILDI! BoxH={box_height}px schrumpft (Max={max_box_height}px) ===")
                            else:
                                forward_desired = 0.10
                    else:
                        # UZAK FAZ: Box boyutuna göre uç
                        if is_shrinking:
                            forward_desired = 0
                            print(f"=== HEDEFE ULAŞILDI! BoxH={box_height}px schrumpft (Max={max_box_height}px) ===")
                        else:
                            forward_desired = 0.15  # Normal hız

                # Hedefi güncelle - 20px tolerans
                current_target = detect_object(camera_data, camera.getWidth(), camera.getHeight(), yolo_model, mode='select')
                if current_target is not None:
                    new_x = current_target['x']
                    new_box = current_target['box']
                    x_diff = abs(new_x - object_x)

                    # SADECE 20px içindeyse güncelle
                    if x_diff < 20:
                        locked_target = {'x': new_x, 'y': current_target['y'], 'box': new_box}
                    else:
                        # Büyük sıçrama - YOKSAY
                        print(f"[UYARI] Target sıçrama yoksayıldı: {object_x:.0f} -> {new_x:.0f} (Fark={x_diff:.0f}px)")

                # Her 10. frame'de log
                if int(mission_timer * 32) % 10 == 0:
                    shrink_status = "SCHRUMPFT!" if is_shrinking else "wächst"
                    phase = "YAKIN(Z+Sensor)" if is_close_phase else "UZAK(X+Y)"
                    print(f"[YAKLAŞMA] {phase}, LockedX={object_x:.0f}, BoxH={box_height}px (Max={max_box_height}, {shrink_status}), Dist={range_front_value:.2f}m, İleri={forward_desired:.2f}, Z_diff={height_diff_desired:.3f}")
            else:
                # Hedef kaybedildi
                print("=== HEDEF KAYBEDİLDİ - GRUP ARAMAYA GERİ DÖN ===")
                mission_state = "SEARCH_GROUP"
                mission_timer = 0
                max_box_height = 0  # Reset

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

        # Sabit yükseklikli PID hız kontrolcüsü
        motor_power = PID_crazyflie.pid(dt, forward_desired, sideways_desired,
                                        yaw_desired, height_desired,
                                        roll, pitch, yaw_rate,
                                        altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global
