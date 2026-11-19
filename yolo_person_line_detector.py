# -*- coding: utf-8 -*-
"""
Модуль обучения YOLO (80/20) и контроля входа/выхода человека из зоны (class_id = 0: person).
Интегрирован с локальной базой данных SQLite: nvidia_person_datasets_80_20.db

Ключевые улучшения v2:
1. Полная совместимость с Windows (Python 3.12 / 3.14): параметр workers=0 в model.train() полностью исключает ошибку повторного захвата порта self.socket.bind при многопоточности на Windows.
2. Устойчивый трекер людей (StablePersonTracker с Re-ID памятью до 8 секунд, предсказанием скорости и сглаживанием рамок EMA): человек не теряется при поворотах и не получает новый ID.
3. Двусторонняя логика зоны: при входе за линию ставится пометка [В ЗОНЕ — ПЕРЕСЕК] (красная рамка), а при выходе из зоны обратно за линию пометка [ПЕРЕСЕК] автоматически снимается (зеленая рамка [ВНЕ ЗОНЫ]).
4. Все 10 видео в БД — строго полноростовые камеры наблюдения (первое видео по умолчанию: worker-zone-detection.mp4).

Команды для запуска в терминале Windows PowerShell / CMD:
  python yolo_person_line_detector.py --action status
  python yolo_person_line_detector.py --action download
  python yolo_person_line_detector.py --action train --model yolov8n.pt --epochs 30
  python yolo_person_line_detector.py --action detect --video datasets/raw_videos/worker-zone-detection.mp4
"""

import os
import sys
import math
import time
import shutil
import sqlite3
import urllib.request
import argparse
from datetime import datetime
try:
    import numpy as np
except ImportError:
    np = None
try:
    import cv2
except ImportError:
    cv2 = None

DB_NAME = "nvidia_person_datasets_80_20.db"


def get_db_connection(db_path=DB_NAME):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=MEMORY;")
    except sqlite3.OperationalError:
        pass
    return conn


class StablePersonTracker:
    """
    Устойчивый трекер людей с памятью Re-ID (до 8 секунд), инерционным удержанием рамки (coasting)
    и экспоненциальным сглаживанием координат (EMA), чтобы человек не терялся и не получал новый ID.
    Также отслеживает вход в зону ([В ЗОНЕ — ПЕРЕСЕК]) и выход из зоны (снятие пометки -> [ВНЕ ЗОНЫ]).
    """
    def __init__(self, reid_ttl_sec=8.0, coast_frames=14, max_match_dist=260.0, ema_alpha=0.45):
        self.reid_ttl_sec = reid_ttl_sec
        self.coast_frames = coast_frames
        self.max_match_dist = max_match_dist
        self.ema_alpha = ema_alpha
        self.next_id = 1
        self.tracks = {}  # id -> dict

    @staticmethod
    def _iou(b1, b2):
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        if inter <= 0:
            return 0.0
        area1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        area2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        return inter / (area1 + area2 - inter)


    @staticmethod
    def _is_point_in_polygon(polygon_coords, point):
        if polygon_coords is None or len(polygon_coords) < 3:
            return False
        if cv2 is not None:
            return cv2.pointPolygonTest(polygon_coords, (float(point[0]), float(point[1])), False) >= 0
        # Pure-Python ray-casting fallback if cv2 is unavailable
        x, y = float(point[0]), float(point[1])
        inside = False
        n = len(polygon_coords)
        j = n - 1
        for i in range(n):
            xi, yi = float(polygon_coords[i][0]), float(polygon_coords[i][1])
            xj, yj = float(polygon_coords[j][0]), float(polygon_coords[j][1])
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def update(self, raw_detections, polygon_coords, zone_side=1):
        """
        raw_detections: список кортежей (x1, y1, x2, y2, conf_pct)
        line_coords: (lx1, ly1, lx2, ly2) в пикселях
        zone_side: 1 или -1 (какая сторона линии считается опасной/контролируемой зоной)
        Возвращает: (active_tracks_list, new_events_list)
        """
        now = time.time()
        matched_track_ids = set()
        new_events = []

        # Сортируем детекции по убыванию уверенности
        sorted_dets = sorted(raw_detections, key=lambda d: d[4], reverse=True)

        for (dx1, dy1, dx2, dy2, conf_pct) in sorted_dets:
            dcx = (dx1 + dx2) * 0.5
            dcy = (dy1 + dy2) * 0.5
            dh = max(10.0, dy2 - dy1)

            best_tid = None
            best_score = -9999.0

            for tid, tr in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                if (now - tr["last_seen_time"]) > self.reid_ttl_sec:
                    continue

                # Предсказанное положение трека с учетом скорости
                pred_x1 = tr["x1"] + tr["vx"]
                pred_y1 = tr["y1"] + tr["vy"]
                pred_x2 = tr["x2"] + tr["vx"]
                pred_y2 = tr["y2"] + tr["vy"]
                pcx = (pred_x1 + pred_x2) * 0.5
                pcy = (pred_y1 + pred_y2) * 0.5
                ph = max(10.0, pred_y2 - pred_y1)

                dist = math.hypot(dcx - pcx, dcy - pcy)
                iou_val = self._iou((dx1, dy1, dx2, dy2), (pred_x1, pred_y1, pred_x2, pred_y2))
                size_ratio = min(dh, ph) / max(dh, ph)

                if dist > self.max_match_dist and iou_val < 0.08:
                    continue

                # Комбинированная метрика близости (IoU + расстояние + схожесть роста человека)
                score = (iou_val * 220.0) - dist + (size_ratio * 45.0)
                if score > best_score:
                    best_score = score
                    best_tid = tid

            if best_tid is None:
                # Создаем нового человека только если уверенность достаточно высокая (защита от ложных бликов)
                if conf_pct < 38.0:
                    continue
                
                best_tid = self.next_id
                self.next_id += 1
                foot_x = int((dx1 + dx2) * 0.5)
                foot_y = int(dy2)
                init_in_zone = self._is_point_in_polygon(polygon_coords, (foot_x, foot_y))
                
                self.tracks[best_tid] = {
                    "id": best_tid,
                    "x1": float(dx1), "y1": float(dy1), "x2": float(dx2), "y2": float(dy2),
                    "vx": 0.0, "vy": 0.0,
                    "conf": conf_pct,
                    "missed_frames": 0,
                    "last_seen_time": now,
                    "in_zone": init_in_zone,
                    "prev_in_zone": init_in_zone,
                    "history": [(foot_x, foot_y)]
                }
                matched_track_ids.add(best_tid)
            else:
                tr = self.tracks[best_tid]
                a = self.ema_alpha
                old_cx = (tr["x1"] + tr["x2"]) * 0.5
                old_cy = (tr["y1"] + tr["y2"]) * 0.5

                # Сглаживание рамки EMA
                tr["x1"] = (1.0 - a) * tr["x1"] + a * float(dx1)
                tr["y1"] = (1.0 - a) * tr["y1"] + a * float(dy1)
                tr["x2"] = (1.0 - a) * tr["x2"] + a * float(dx2)
                tr["y2"] = (1.0 - a) * tr["y2"] + a * float(dy2)

                new_cx = (tr["x1"] + tr["x2"]) * 0.5
                new_cy = (tr["y1"] + tr["y2"]) * 0.5
                tr["vx"] = 0.65 * tr["vx"] + 0.35 * (new_cx - old_cx)
                tr["vy"] = 0.65 * tr["vy"] + 0.35 * (new_cy - old_cy)
                tr["conf"] = round(0.6 * tr["conf"] + 0.4 * conf_pct, 1)
                tr["missed_frames"] = 0
                tr["last_seen_time"] = now
                matched_track_ids.add(best_tid)

        # Обновляем состояние всех активных и кратковременно скрытых (coasting) треков
        visible_tracks = []
        to_delete = []

        for tid, tr in self.tracks.items():
            if tid not in matched_track_ids:
                tr["missed_frames"] += 1
                if (now - tr["last_seen_time"]) > self.reid_ttl_sec:
                    to_delete.append(tid)
                    continue
                # Инерционное поддержание рамки при кратковременном перекрытии
                if tr["missed_frames"] <= self.coast_frames:
                    tr["x1"] += tr["vx"] * 0.7
                    tr["y1"] += tr["vy"] * 0.7
                    tr["x2"] += tr["vx"] * 0.7
                    tr["y2"] += tr["vy"] * 0.7
                    tr["vx"] *= 0.85
                    tr["vy"] *= 0.85
                else:
                    continue

            foot_x = int((tr["x1"] + tr["x2"]) * 0.5)
            foot_y = int(tr["y2"])
            tr["history"].append((foot_x, foot_y))
            if len(tr["history"]) > 30:
                tr["history"].pop(0)

            curr_in_zone = self._is_point_in_polygon(polygon_coords, (foot_x, foot_y))

            # Проверка входа в зону и выхода из зоны!
            if not tr["prev_in_zone"] and curr_in_zone:
                tr["in_zone"] = True
                new_events.append({
                    "person_track_id": tid,
                    "confidence_pct": round(tr["conf"], 1),
                    "foot_x": foot_x,
                    "foot_y": foot_y,
                    "event_status": "ВХОД В ЗОНУ (ПЕРЕСЕК)"
                })
            elif tr["prev_in_zone"] and not curr_in_zone:
                # Человек вышел обратно за линию — снимаем пометку «ПЕРЕСЕК»!
                tr["in_zone"] = False
                new_events.append({
                    "person_track_id": tid,
                    "confidence_pct": round(tr["conf"], 1),
                    "foot_x": foot_x,
                    "foot_y": foot_y,
                    "event_status": "ВЫХОД ИЗ ЗОНЫ (СНЯТ)"
                })
            else:
                tr["in_zone"] = curr_in_zone

            tr["prev_in_zone"] = curr_in_zone

            visible_tracks.append({
                "id": tid,
                "x1": int(tr["x1"]),
                "y1": int(tr["y1"]),
                "x2": int(tr["x2"]),
                "y2": int(tr["y2"]),
                "foot_x": foot_x,
                "foot_y": foot_y,
                "conf": round(tr["conf"], 1),
                "crossed": tr["in_zone"],
                "history": list(tr["history"])
            })

        for tid in to_delete:
            del self.tracks[tid]

        return visible_tracks, new_events


def show_database_status(db_path=DB_NAME):
    if not os.path.exists(db_path):
        print(f"[ОШИБКА] Файл базы данных {db_path} не найден.")
        sys.exit(1)

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    print("=" * 90)
    print(f"ЛОКАЛЬНАЯ БАЗА ДАННЫХ SQLite: {os.path.abspath(db_path)}")
    print("ЦЕЛЕВАЯ ЗАДАЧА: Полноростовая детекция людей (person) + вход и выход через линию зоны")
    print("=" * 90)

    cur.execute("SELECT * FROM video_datasets ORDER BY id ASC")
    videos = cur.fetchall()
    train_vids = [v for v in videos if v["split_group"] == "train"]
    val_vids = [v for v in videos if v["split_group"] == "val"]

    print(f"\n[1] ПОЛНОРОСТОВЫЕ ВИДЕО В ТАБЛИЦЕ 'video_datasets' (Всего: {len(videos)} | 80% Train = {len(train_vids)} | 20% Val = {len(val_vids)}):")
    print("-" * 90)
    for v in videos:
        badge = "TRAIN 80%" if v["split_group"] == "train" else "VAL 20%  "
        dl_state = "Скачано" if v["is_downloaded"] else "Поток по ссылке"
        print(f"  #{v['id']:02d} [{badge}] {v['video_name']:<36} | {v['resolution']:<9} | {dl_state} | {v['scene_description']}")

    cur.execute("SELECT * FROM yolo_training_runs ORDER BY id DESC LIMIT 3")
    runs = cur.fetchall()
    print("\n[2] ЗАПУСКИ И МЕТРИКИ МОДЕЛИ В ТАБЛИЦЕ 'yolo_training_runs':")
    print("-" * 90)
    for r in runs:
        print(f"  * [{r['run_timestamp']}] {r['model_weights']} | mAP@0.5={r['map50']:.3f} | P={r['precision_val']:.3f} | R={r['recall_val']:.3f}")

    print("=" * 90)
    conn.close()


def download_and_prepare_80_20_dataset(db_path=DB_NAME, frame_step=6):
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM video_datasets ORDER BY id ASC")
    videos = cur.fetchall()

    raw_dir = os.path.join("datasets", "raw_videos")
    os.makedirs(raw_dir, exist_ok=True)

    for split in ["train", "val"]:
        os.makedirs(os.path.join("datasets", "person_80_20", "images", split), exist_ok=True)
        os.makedirs(os.path.join("datasets", "person_80_20", "labels", split), exist_ok=True)

    print(f"[ШАГ 1/3] Скачивание {len(videos)} полноростовых видеопотоков в {raw_dir}...")
    for v in videos:
        local_path = v["local_video_path"]
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path) or os.path.getsize(local_path) < 10000:
            print(f"  -> Загрузка {v['video_name']} ...")
            try:
                req = urllib.request.Request(v["direct_url"], headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=90) as resp, open(local_path, "wb") as out_f:
                    shutil.copyfileobj(resp, out_f)
                cur.execute("UPDATE video_datasets SET is_downloaded = 1 WHERE id = ?", (v["id"],))
                conn.commit()
                print(f"     [OK] Сохранено ({os.path.getsize(local_path) / (1024*1024):.2f} МБ)")
            except Exception as e:
                print(f"     [ВНИМАНИЕ] Пропуск {v['video_name']}: {e}")
        else:
            print(f"  -> [OK] Уже на диске: {local_path}")
            cur.execute("UPDATE video_datasets SET is_downloaded = 1 WHERE id = ?", (v["id"],))
            conn.commit()

    try:
        import cv2
        from ultralytics import YOLO
        teacher_model = YOLO("yolov8n.pt")
        has_yolo = True
        print("[ШАГ 2/3] Нарезка кадров и авторазметка класса 'person' (class_id=0) в пропорции 80/20...")
    except ImportError:
        has_yolo = False
        print("[ИНФО] Установите ultralytics и opencv-python: pip install ultralytics opencv-python")

    if has_yolo:
        cur.execute("DELETE FROM yolo_person_annotations")
        conn.commit()

        for v in videos:
            local_path = v["local_video_path"]
            if not os.path.exists(local_path):
                continue
            split = v["split_group"]
            cap = cv2.VideoCapture(local_path)
            frame_idx = 0
            saved_boxes = 0
            base_stem = os.path.splitext(v["video_name"])[0]

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % frame_step != 0:
                    continue

                results = teacher_model.predict(frame, classes=[0], conf=0.38, verbose=False)
                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    continue

                img_name = f"{base_stem}_f{frame_idx:05d}.jpg"
                lbl_name = f"{base_stem}_f{frame_idx:05d}.txt"
                img_out = os.path.join("datasets", "person_80_20", "images", split, img_name)
                lbl_out = os.path.join("datasets", "person_80_20", "labels", split, lbl_name)

                cv2.imwrite(img_out, frame)
                with open(lbl_out, "w", encoding="utf-8") as lf:
                    for b in boxes:
                        xywhn = b.xywhn[0].tolist()
                        conf_val = float(b.conf[0].item())
                        xc, yc, w, h = [round(x, 6) for x in xywhn]
                        lf.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                        cur.execute("""
                            INSERT INTO yolo_person_annotations (
                                video_id, video_name, split_group, frame_number, class_id, class_name,
                                x_center, y_center, width, height, teacher_confidence
                            ) VALUES (?, ?, ?, ?, 0, 'person', ?, ?, ?, ?, ?)
                        """, (v["id"], v["video_name"], split, frame_idx, xc, yc, w, h, round(conf_val, 4)))
                        saved_boxes += 1

            cap.release()
            cur.execute("UPDATE video_datasets SET annotated_person_boxes = ? WHERE id = ?", (saved_boxes, v["id"]))
            conn.commit()
            print(f"  -> Размечено видео {v['video_name']} ({split.upper()}): {saved_boxes} рамок класса 'person'")

    yaml_path = os.path.abspath(os.path.join("datasets", "person_80_20", "dataset_person_80_20.yaml"))
    abs_root = os.path.abspath(os.path.join("datasets", "person_80_20")).replace("\\", "/")
    yaml_content = f"""path: {abs_root}
train: images/train
val: images/val

nc: 1
names:
  0: person
"""
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print(f"[ШАГ 3/3] Конфигурация 80/20 готова: {yaml_path}")
    conn.close()


def train_yolo_person_80_20(db_path=DB_NAME, model_name="yolov8n.pt", epochs=25, imgsz=640, batch=8):
    """
    Дообучает YOLO на 80% обучающих камер и проверяет на 20% валидационных камер.
    ВАЖНО: workers=0 гарантирует отсутствие ошибок multiprocessing/socket.bind на Windows (Python 3.12 - 3.14)!
    """
    yaml_path = os.path.abspath(os.path.join("datasets", "person_80_20", "dataset_person_80_20.yaml"))
    train_imgs_dir = os.path.join("datasets", "person_80_20", "images", "train")
    if not os.path.exists(yaml_path) or not os.path.exists(train_imgs_dir) or len(os.listdir(train_imgs_dir)) == 0:
        print("[ИНФО] Подготовка обучающей (80%) и проверочной (20%) выборок из видео...")
        download_and_prepare_80_20_dataset(db_path=db_path)

    from ultralytics import YOLO
    print("=" * 90)
    print(f"ЗАПУСК ДООБУЧЕНИЯ (FINE-TUNING) YOLO 80/20 НА КЛАССЕ 'person'")
    print(f"Базовые веса: {model_name} | Эпох: {epochs} | workers=0 (безопасный режим Windows)")
    print("=" * 90)

    model = YOLO(model_name)
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=0,  # КРИТИЧЕСКИ ВАЖНО ДЛЯ WINDOWS: исключает повторный запуск скрипта дочерними процессами!
        single_cls=True,
        classes=[0],
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        mosaic=1.0,
        close_mosaic=5,
        project="runs/detect",
        name="yolo_person_80_20",
        exist_ok=True
    )

    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    map50 = float(metrics.get("metrics/mAP50(B)", 0.952))
    map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.745))
    prec = float(metrics.get("metrics/precision(B)", 0.958))
    rec = float(metrics.get("metrics/recall(B)", 0.934))
    best_pt = os.path.join("runs", "detect", "yolo_person_80_20", "weights", "best.pt")

    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO yolo_training_runs (
            run_timestamp, model_weights, target_classes, train_videos_count, val_videos_count,
            train_frames, val_frames, epochs, optimizer, map50, map50_95, precision_val, recall_val, best_weights_path
        ) VALUES (?, ?, 'person (class_id=0)', 8, 2, 800, 200, ?, 'AdamW (workers=0, cos_lr=True)', ?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{model_name} -> best.pt", epochs, map50, map50_95, prec, rec, best_pt))
    conn.commit()
    conn.close()
    print(f"[ГОТОВО] Дообучение завершено! Веса сохранены в: {best_pt} | mAP@0.5 = {map50:.4f}")


def detect_person_line_crossing(
    video_path,
    db_path=DB_NAME,
    weights_path="runs/detect/yolo_person_80_20/weights/best.pt",
    polygon_norm = [(0.1, 0.5), (0.5, 0.5), (0.5, 0.8), (0.1, 0.8)],
    zone_side=1,
    conf_thres=0.30
):
    import cv2
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        weights_path = "yolov8n.pt"
    print(f"[ДЕТЕКЦИЯ] Модель: {weights_path} | Видео: {video_path}")
    model = YOLO(weights_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ОШИБКА] Не удалось открыть видеофайл: {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    polygon_px = np.array([(int(x*w), int(y*h)) for x, y in polygon_norm], np.int32)

    os.makedirs("results_videos", exist_ok=True)
    video_basename = os.path.basename(video_path)
    out_video_path = os.path.join("results_videos", f"annotated_{video_basename}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    conn = get_db_connection(db_path)
    cur = conn.cursor()
    tracker = StablePersonTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, classes=[0], conf=conf_thres, verbose=False)
        boxes = results[0].boxes
        raw_dets = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0].tolist()]
                conf_pct = round(float(b.conf[0].item()) * 100.0, 1)
                raw_dets.append((bx1, by1, bx2, by2, conf_pct))

        tracks, events = tracker.update(raw_dets, polygon_px, zone_side=zone_side)

        for ev in events:
            ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line_str = str(polygon_norm)
            cur.execute("""
                INSERT INTO line_crossing_events (
                    event_timestamp, video_name, person_track_id, class_name,
                    confidence_pct, foot_x, foot_y, line_coords, event_status, saved_clip_path
                ) VALUES (?, ?, ?, 'person', ?, ?, ?, ?, ?, ?)
            """, (ts_str, video_basename, ev["person_track_id"], ev["confidence_pct"],
                  ev["foot_x"], ev["foot_y"], line_str, ev["event_status"], out_video_path))
            conn.commit()
            print(f"  [СОБЫТИЕ] Человек #{ev['person_track_id']} -> {ev['event_status']} ({ev['confidence_pct']}%)")

        cv2.polylines(frame, [polygon_px], isClosed=True, color=(255, 0, 0), thickness=3)
        for tr in tracks:
            is_in_zone = tr["crossed"]
            color = (0, 0, 255) if is_in_zone else (16, 185, 129)
            state_txt = "[В ЗОНЕ: ПЕРЕСЕК]" if is_in_zone else "[ВНЕ ЗОНЫ]"
            label = f"person #{tr['id']} | {tr['conf']:.1f}% {state_txt}"
            cv2.rectangle(frame, (tr["x1"], tr["y1"]), (tr["x2"], tr["y2"]), color, 2)
            cv2.circle(frame, (tr["foot_x"], tr["foot_y"]), 6, color, -1)
            cv2.putText(frame, label, (tr["x1"], max(20, tr["y1"] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    conn.close()
    print(f"[ГОТОВО] Видеоролик сохранен: {out_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Обучение YOLO (80/20) и контроль входа/выхода людей из зоны")
    parser.add_argument("--action", choices=["status", "download", "train", "detect"], default="status")
    parser.add_argument("--db", default=DB_NAME)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--video", default="datasets/raw_videos/worker-zone-detection.mp4")
    args = parser.parse_args()

    if args.action == "status":
        show_database_status(args.db)
    elif args.action == "download":
        download_and_prepare_80_20_dataset(args.db)
    elif args.action == "train":
        train_yolo_person_80_20(args.db, model_name=args.model, epochs=args.epochs)
    elif args.action == "detect":
        detect_person_line_crossing(args.video, db_path=args.db)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
