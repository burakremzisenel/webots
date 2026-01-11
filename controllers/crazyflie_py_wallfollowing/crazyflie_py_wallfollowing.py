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

def detect_object(image_data, width, height, model, target_color='red', skip_color_filter=False):
    """
    BASİT: Görüntüdeki kırmızı nesneleri bulur.
    Ekran merkezine en yakın nesneyi döndürür.
    """
    if image_data is None:
        return None

    # Görüntüyü dönüştür
    img_array = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
    img_bgr = np.array(img_array[:, :, :3])

    # YOLO Algılama
    results = model(img_bgr, verbose=False, stream=True)

    red_candidates = []

    for result in results:
        for box in result.boxes:
            # Sadece şişe, bardak, vazo
            if int(box.cls) in [39, 41, 75]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Renk kontrolü (skip_color_filter=True değilse her zaman aktif)
                is_red = skip_color_filter  # Skip ise otomatik geçerli

                if not skip_color_filter:
                    # ROI çıkar
                    roi = img_bgr[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]

                    if roi.size > 0:
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        mask1 = cv2.inRange(hsv_roi, np.array([0, 50, 20]), np.array([10, 255, 255]))
                        mask2 = cv2.inRange(hsv_roi, np.array([170, 50, 20]), np.array([180, 255, 255]))
                        red_ratio = cv2.countNonZero(mask1 + mask2) / (roi.shape[0] * roi.shape[1])

                        if red_ratio > 0.1:
                            is_red = True

                if is_red:
                    cx = float(box.xywh[0][0])
                    cy = float(box.xywh[0][1])
                    red_candidates.append({'x': cx, 'y': cy, 'box': (x1, y1, x2, y2)})
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Sarı
                else:
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Mavi (kırmızı değil)

    # Ekran merkezine EN YAKIN nesneyi seç
    target = None
    if red_candidates:
        center_x = width / 2
        # Ekran merkezine olan mesafeye göre sırala
        red_candidates.sort(key=lambda c: abs(c['x'] - center_x))
        target = red_candidates[0]

        # Hedefi yeşil işaretle
        tx1, ty1, tx2, ty2 = target['box']
        cv2.rectangle(img_bgr, (tx1, ty1), (tx2, ty2), (0, 255, 0), 3)
        cv2.putText(img_bgr, "TARGET", (tx1, ty1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Sichtfeld", img_bgr)
    cv2.waitKey(1)

    if target:
        return target['x'], target['y'], target['box']
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

    # Görev Durumu: BASİT
    mission_state = "TAKEOFF"  # TAKEOFF -> SEARCH -> APPROACH
    mission_timer = 0.0
    locked_target = None  # Bir hedef bulduğumuzda, ona kilitleniyoruz!

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

        # --- BASİT GÖREV MANTIĞI ---
        camera_data = camera.getImage()
        range_front_value = range_front.getValue() / 1000
        range_right_value = range_right.getValue() / 1000
        range_left_value = range_left.getValue() / 1000

        # Nesne algılama (her zaman aktif)
        object_coords = detect_object(camera_data, camera.getWidth(), camera.getHeight(), yolo_model)

        if mission_state == "TAKEOFF":
            # 2 saniye yüksel
            if mission_timer > 2.0:
                print("=== KALKIŞ TAMAMLANDI - ARAMAYA BAŞLA ===")
                mission_state = "SEARCH"
                mission_timer = 0

        elif mission_state == "SEARCH":
            # Yavaşça dön ve kırmızı nesne ara
            yaw_desired = 0.3  # Yumuşak dönüş

            if object_coords is not None:
                object_x, object_y, object_box = object_coords
                # Bu hedefe KİLİTLEN!
                locked_target = {'x': object_x, 'y': object_y, 'box': object_box}
                print(f"=== HEDEF KİLİTLENDİ X={object_x:.0f} - YAKLAŞMAYA BAŞLA ===")
                mission_state = "APPROACH"
            elif mission_timer > 20.0:  # 20 saniye arandı
                print("=== ARAMA ZAMAN AŞIMI - DEVAM ET ===")
                mission_timer = 0

        elif mission_state == "APPROACH":
            # ÖNEMLİ: SADECE locked_target kullan, object_coords değil!
            # Bu, kutular arasında zıplamayı önler
            if locked_target is not None:
                object_x = locked_target['x']
                object_y = locked_target['y']

                center_x = camera.getWidth() / 2
                error_x = object_x - center_x

                # Nesneye dön
                yaw_desired = -0.004 * error_x
                yaw_desired = max(-0.5, min(0.5, yaw_desired))  # Sınırla

                # Merkezlenmişse ileri uç
                if abs(error_x) < 80 and range_front_value > 0.5:
                    forward_desired = 0.1  # Yavaş ileri

                # locked_target'i SADECE yeni nesne çok yakınsa güncelle (zıplama yok!)
                if object_coords is not None:
                    new_x, new_y, new_box = object_coords
                    # Sadece < 30px sapma varsa güncelle (aynı kutu)
                    if abs(new_x - object_x) < 30:
                        locked_target = {'x': new_x, 'y': new_y, 'box': new_box}
                    # Değilse: Eski hedefi tut (zıplamaları yoksay)

                print(f"[YAKLAŞMA] X={object_x:.0f}, Hata={error_x:.0f}, Mesafe={range_front_value:.2f}m, Yaw={yaw_desired:.2f}")
            else:
                # Hedef kaybedildi
                print("=== HEDEF KAYBEDİLDİ - ARAMAYA GERİ DÖN ===")
                locked_target = None
                mission_state = "SEARCH"
                mission_timer = 0

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
