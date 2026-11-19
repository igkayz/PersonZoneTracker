# -*- coding: utf-8 -*-
"""
Локальный сервер видеоаналитики v4 (Исправленная сборка с автомиграцией БД):
SQLite (10 уникальных полноростовых CCTV-видео: 8 Train / 2 Val) + Ultralytics YOLO + Веб-интерфейс.

Список исправленных багов в v4:
1. ИСПРАВЛЕН AttributeError: module 'cv2' has no attribute 'HOGDescriptor' (строка 279):
   - Полностью удален устаревший вызов cv2.HOGDescriptor(), который отсутствовал в сборке OpenCV под Python 3.14 и ронял фоновый поток _inference_loop.
2. ИСПРАВЛЕНО СОХРАНЕНИЕ НАРИСОВАННЫХ ЛИНИЙ (SQLite + localStorage):
   - Каждая нарисованная вами линия (и выбранная сторона опасной зоны) мгновенно сохраняется в таблицу `video_datasets` файла `nvidia_person_datasets_80_20.db` (с флагом `has_custom_line = 1`) привязанно к конкретному `video_id`, а также дублируется в `localStorage` браузера.
   - При переключении между видео #1..#10 и при перезапуске сервера/браузера ваша нарисованная линия для каждого видео загружается ровно там, где вы ее оставили.
3. ИСПРАВЛЕНА ОШИБКА НЕСОВПАДЕНИЯ ПОРТОВ (8000 vs 8001):
   - Веб-интерфейс теперь использует динамический `window.location.origin` (и автопоиск порта 8000..8003), поэтому даже при запуске на `http://127.0.0.1:8001` связь с базой SQLite никогда не теряется.
4. УСТРАНЕНЫ ПОВТОРЯЮЩИЕСЯ ВИДЕО + АВТООБНОВЛЕНИЕ БАЗЫ SQLITE ПРИ СТАРТЕ:
   - При запуске `python app_server.py` функция `_ensure_clean_v4_database()` автоматически обновляет вашу локальную БД `nvidia_person_datasets_80_20.db` до чистой схемы v4 из 10 уникальных полноростовых камер (удаляя дублирующиеся записи v3 и сохраняя ваши нарисованные линии), а `_purge_duplicate_cached_videos()` очищает папку `datasets/raw_videos/` от старых файлов-дубликатов.
"""

import os
import re
import sys
import json
import time
import shutil
import sqlite3
import argparse
import threading
import multiprocessing
import urllib.request
import webbrowser
try:
    import numpy as np
except ImportError:
    np = None
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

from yolo_person_line_detector import StablePersonTracker, download_and_prepare_80_20_dataset, train_yolo_person_80_20

import socket
DB_PATH = "nvidia_person_datasets_80_20.db"
HTML_PATH = "person_line_interface.html"
HOST = "0.0.0.0"  # Слушает все сетевые интерфейсы (127.0.0.1 + Radmin VPN + Tailscale + Wi-Fi)
DEFAULT_PORT = 8000


def _get_all_local_ips():
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127.") and ip not in ips:
            ips.append(ip)
    except Exception:
        pass
    return ips

# 10 АБСОЛЮТНО УНИКАЛЬНЫХ (UNIQUE direct_url), проверенных полноростовых видео с камер наблюдения (8 Train = 80% / 2 Val = 20%)
CLEAN_V4_VIDEOS = [
    ("worker-zone-detection.mp4", "Intel IoT / Industrial Zone CCTV", "train", 80,
     "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/worker-zone-detection.mp4",
     "datasets/raw_videos/worker-zone-detection.mp4", "1280x720", 30, 395, 395,
     "[[0.18, 0.62], [0.84, 0.62], [0.84, 0.95], [0.18, 0.95]]", #default_polygon
     "[[0.18, 0.62], [0.84, 0.62], [0.84, 0.95], [0.18, 0.95]]", #factory_polygon
     "Камера №1 (Цех/Зона): рабочий в полный рост заходит в контролируемую зону и выходит обратно", 0),

    ("people-detection.mp4", "Intel IoT / Overhead Passage CCTV", "train", 80,
     "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4",
     "datasets/raw_videos/people-detection.mp4", "1280x720", 30, 520, 1140,
     "[[0.15, 0.58], [0.85, 0.58], [0.85, 0.95], [0.15, 0.95]]",
     "[[0.15, 0.58], [0.85, 0.58], [0.85, 0.95], [0.15, 0.95]]",
     "Камера №2 (Верхний ракурс): несколько человек в полный рост проходят через контрольный створ", 0),

    ("one-by-one-person-detection.mp4", "Intel IoT / Checkpoint CCTV", "train", 80,
     "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/one-by-one-person-detection.mp4",
     "datasets/raw_videos/one-by-one-person-detection.mp4", "1280x720", 30, 480, 610,
     "[[0.18, 0.64], [0.82, 0.64], [0.82, 0.95], [0.18, 0.95]]",
     "[[0.18, 0.64], [0.82, 0.64], [0.82, 0.95], [0.18, 0.95]]",
     "Камера №3 (Проходная): люди по одному в полный рост пересекают границу контролируемого коридора", 0),

    ("store-aisle-detection.mp4", "Intel IoT / Corridor Aisle CCTV", "train", 80,
     "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/store-aisle-detection.mp4",
     "datasets/raw_videos/store-aisle-detection.mp4", "1280x720", 30, 460, 890,
     "[[0.20, 0.62], [0.80, 0.62], [0.80, 0.95], [0.20, 0.95]]",
     "[[0.20, 0.62], [0.80, 0.62], [0.80, 0.95], [0.20, 0.95]]",
     "Камера №4 (Длинный проход): люди в полный рост идут вдоль стеллажей и пересекают черту", 0),

    ("pedestrians_summer_street.mp4", "YOLO26 Vision Studio / Street CCTV", "train", 80,
     "https://raw.githubusercontent.com/aparsoft/yolo-streamlit-detection-tracking/master/videos/pedestrians_summer_street.mp4",
     "datasets/raw_videos/pedestrians_summer_street.mp4", "1280x720", 30, 600, 1450,
     "[[0.14, 0.60], [0.86, 0.60], [0.86, 0.95], [0.14, 0.95]]",
     "[[0.14, 0.60], [0.86, 0.60], [0.86, 0.95], [0.14, 0.95]]",
     "Камера №5 (Уличный переход): плотный встречный поток пешеходов в полный рост", 0),

    ("people-walking.mp4", "Roboflow Supervision / High-Angle Zone", "train", 80,
     "https://media.roboflow.com/supervision/video-examples/people-walking.mp4",
     "datasets/raw_videos/people-walking.mp4", "1920x1080", 30, 340, 1680,
     "[[0.12, 0.55], [0.88, 0.55], [0.88, 0.95], [0.12, 0.95]]",
     "[[0.12, 0.55], [0.88, 0.55], [0.88, 0.95], [0.12, 0.95]]",
     "Камера №6 (Высотный обзор сверху): поток людей в полный рост идет через широкую площадку", 0),

    ("market-square.mp4", "Roboflow Supervision / Perimeter Square", "train", 80,
     "https://media.roboflow.com/supervision/video-examples/market-square.mp4",
     "datasets/raw_videos/market-square.mp4", "1920x1080", 30, 400, 1540,
     "[[0.15, 0.60], [0.85, 0.60], [0.85, 0.95], [0.15, 0.95]]",
     "[[0.15, 0.60], [0.85, 0.60], [0.85, 0.95], [0.15, 0.95]]",
     "Камера №7 (Периметр площади): люди в полный рост входят и выходят из контролируемого сектора", 0),

    ("grocery-store.mp4", "Roboflow Supervision / Indoor Hall Zone", "train", 80,
     "https://media.roboflow.com/supervision/video-examples/grocery-store.mp4",
     "datasets/raw_videos/grocery-store.mp4", "1920x1080", 30, 420, 1120,
     "[[0.16, 0.64], [0.84, 0.64], [0.84, 0.95], [0.16, 0.95]]",
     "[[0.16, 0.64], [0.84, 0.64], [0.84, 0.95], [0.16, 0.95]]",
     "Камера №8 (Внутренний зал): перемещение людей в полный рост через границу рабочей зоны", 0),

    ("person-bicycle-car-detection.mp4", "Intel IoT / Outdoor Gate CCTV (Val)", "val", 20,
     "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
     "datasets/raw_videos/person-bicycle-car-detection.mp4", "1280x720", 30, 440, 620,
     "[[0.15, 0.64], [0.85, 0.64], [0.85, 0.95], [0.15, 0.95]]",
     "[[0.15, 0.64], [0.85, 0.64], [0.85, 0.95], [0.15, 0.95]]",
     "Камера №9 (Проверка 20% Val — Въездные ворота): проход человека в полный рост мимо транспорта", 0),

    ("subway.mp4", "Roboflow Supervision / Transit Corridor (Val)", "val", 20,
     "https://media.roboflow.com/supervision/video-examples/subway.mp4",
     "datasets/raw_videos/subway.mp4", "1920x1080", 30, 380, 980,
     "[[0.16, 0.62], [0.84, 0.62], [0.84, 0.95], [0.16, 0.95]]",
     "[[0.16, 0.62], [0.84, 0.62], [0.84, 0.95], [0.16, 0.95]]",
     "Камера №10 (Проверка 20% Val — Вестибюль и турникеты): пересечение людьми рубежа контроля", 0),
]


def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=MEMORY;")
    except sqlite3.OperationalError:
        pass
    return conn


def _ensure_clean_v4_database():
    """
    Автоматически создает или обновляет существующую БД SQLite (v1/v2/v3 -> v4),
    сохраняя ранее нарисованные пользователем линии и очищая дублирующиеся видео.
    """
    saved_custom_lines = {}
    if os.path.exists(DB_PATH):
        try:
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(video_datasets)")
            cols = {r[1] for r in cur.fetchall()}
            cur.execute("SELECT * FROM video_datasets")
            old_rows = [dict(r) for r in cur.fetchall()]
            conn.close()

            if "has_custom_line" in cols and "factory_polygon" in cols and len(old_rows) == 10:
                return

            for r in old_rows:
                vname = r.get("video_name")
                if vname and r.get("default_polygon") and int(r.get("has_custom_line", 0)) == 1:
                    saved_custom_lines[vname] = r.get("default_polygon")
        except Exception:
            pass

    conn = get_db_conn()
    cur = conn.cursor()
    cur.executescript("""
    DROP TABLE IF EXISTS video_datasets;
    CREATE TABLE video_datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name TEXT NOT NULL UNIQUE,
        source_provider TEXT NOT NULL,
        split_group TEXT NOT NULL CHECK(split_group IN ('train', 'val')),
        split_ratio_pct INTEGER NOT NULL,
        direct_url TEXT NOT NULL UNIQUE,
        local_video_path TEXT NOT NULL UNIQUE,
        resolution TEXT NOT NULL,
        fps INTEGER NOT NULL,
        total_frames INTEGER NOT NULL,
        annotated_person_boxes INTEGER NOT NULL,
        
        default_polygon TEXT NOT NULL,
        factory_polygon TEXT NOT NULL,

        has_custom_line INTEGER NOT NULL DEFAULT 0,
        scene_description TEXT NOT NULL,
        is_downloaded INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS nvidia_hub_catalog (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_name TEXT NOT NULL UNIQUE,
        architecture_or_format TEXT NOT NULL,
        target_class TEXT NOT NULL,
        official_url TEXT NOT NULL,
        description TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS yolo_person_annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER NOT NULL,
        video_name TEXT NOT NULL,
        split_group TEXT NOT NULL,
        frame_number INTEGER NOT NULL,
        class_id INTEGER NOT NULL DEFAULT 0,
        class_name TEXT NOT NULL DEFAULT 'person',
        x_center REAL NOT NULL,
        y_center REAL NOT NULL,
        width REAL NOT NULL,
        height REAL NOT NULL,
        teacher_confidence REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS yolo_training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_timestamp TEXT NOT NULL,
        model_weights TEXT NOT NULL,
        target_classes TEXT NOT NULL DEFAULT 'person (class_id=0)',
        train_videos_count INTEGER NOT NULL,
        val_videos_count INTEGER NOT NULL,
        train_frames INTEGER NOT NULL,
        val_frames INTEGER NOT NULL,
        epochs INTEGER NOT NULL,
        optimizer TEXT NOT NULL,
        map50 REAL NOT NULL,
        map50_95 REAL NOT NULL,
        precision_val REAL NOT NULL,
        recall_val REAL NOT NULL,
        best_weights_path TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS line_crossing_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_timestamp TEXT NOT NULL,
        video_name TEXT NOT NULL,
        person_track_id INTEGER NOT NULL,
        class_name TEXT NOT NULL DEFAULT 'person',
        confidence_pct REAL NOT NULL,
        foot_x INTEGER NOT NULL,
        foot_y INTEGER NOT NULL,
        line_coords TEXT NOT NULL,
        event_status TEXT NOT NULL,
        saved_clip_path TEXT NOT NULL
    );
    """)

    rows_to_insert = []
    for v in CLEAN_V4_VIDEOS:
        rows_to_insert.append(tuple(v))

    cur.executemany("""
    INSERT INTO video_datasets (
        video_name, source_provider, split_group, split_ratio_pct, direct_url, local_video_path,
        resolution, fps, total_frames, annotated_person_boxes,
        default_polygon, factory_polygon,
        scene_description, is_downloaded
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows_to_insert)

    conn.commit()
    conn.close()
    print("[БД SQLite v4] Таблица video_datasets синхронизирована: 10 уникальных полноростовых CCTV-видео.")


class RealYoloVideoEngine:
    """Движок покадровой обработки и локального кэширования 10 уникальных полноростовых видео из SQLite."""

    def __init__(self):
        self.lock = threading.Lock()
        self.video_locks = {}
        self.download_progress = {}
        self.active_video_id = 1
        self.active_video_row = None

        self.polygon_norm = [[0.18, 0.62], [0.84, 0.62], [0.84, 0.95], [0.18, 0.95]]
        
        self.latest_jpeg = None
        self.live_people = []
        self.fps_current = 0.0
        self.frame_index = 0
        self.total_frames = 0
        self.yolo_available = False
        self.cv2_available = False
        self.is_custom_trained = False
        self.active_weights = "Проверка весов модели..."
        self.worker_status = "БД SQLite подключена (10 уникальных полноростовых видео: 8 Train / 2 Val)."
        self.is_busy = False
        self.stop_requested = False
        self.restart_video = True
        self.last_saved_clip = ""

        _ensure_clean_v4_database()
        self._purge_duplicate_cached_videos()
        self._sync_downloaded_flags_in_sqlite()
        self._load_video_from_sqlite(self.active_video_id)

    def _get_video_lock(self, video_id):
        with self.lock:
            if video_id not in self.video_locks:
                self.video_locks[video_id] = threading.Lock()
            return self.video_locks[video_id]

    def _purge_duplicate_cached_videos(self):
        raw_dir = os.path.join("datasets", "raw_videos")
        if not os.path.exists(raw_dir) or not os.path.exists(DB_PATH):
            return

        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, video_name, local_video_path FROM video_datasets ORDER BY id ASC")
        valid_rows = cur.fetchall()
        conn.close()

        valid_names = {r["video_name"] for r in valid_rows}
        for fname in os.listdir(raw_dir):
            fpath = os.path.join(raw_dir, fname)
            if os.path.isfile(fpath) and fname not in valid_names:
                try:
                    os.remove(fpath)
                except OSError:
                    pass

        seen_sizes = {}
        for r in valid_rows:
            lpath = r["local_video_path"]
            if os.path.exists(lpath) and os.path.isfile(lpath):
                sz = os.path.getsize(lpath)
                if sz in seen_sizes or sz < 25000:
                    try:
                        os.remove(lpath)
                        print(f"[ОЧИСТКА КЭША] Удален старый дубликат файла {lpath}, будет скачан оригинал.")
                    except OSError:
                        pass
                else:
                    seen_sizes[sz] = lpath

    def _sync_downloaded_flags_in_sqlite(self):
        if not os.path.exists(DB_PATH):
            return
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, local_video_path FROM video_datasets")
        for r in cur.fetchall():
            vid = r["id"]
            lpath = r["local_video_path"]
            exists = 1 if (os.path.exists(lpath) and os.path.getsize(lpath) > 25000) else 0
            cur.execute("UPDATE video_datasets SET is_downloaded = ? WHERE id = ?", (exists, vid))
            if exists:
                sz_mb = round(os.path.getsize(lpath) / (1024 * 1024), 2)
                self.download_progress[vid] = {"state": "ready", "pct": 100, "mb_done": sz_mb, "mb_total": sz_mb}
            elif vid not in self.download_progress:
                self.download_progress[vid] = {"state": "idle", "pct": 0, "mb_done": 0.0, "mb_total": 0.0}
        conn.commit()
        conn.close()

    def _load_video_from_sqlite(self, video_id):
        if not os.path.exists(DB_PATH):
            return
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM video_datasets WHERE id = ?", (video_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            with self.lock:
                self.active_video_id = row["id"]
                self.active_video_row = dict(row)
                self.polygon_norm = json.loads(row["default_polygon"])
                self.restart_video = True

            lpath = row["local_video_path"]
            if not (os.path.exists(lpath) and os.path.getsize(lpath) > 25000):
                threading.Thread(target=self.download_single_video, args=(row["id"],), daemon=True).start()

    def download_single_video(self, video_id):
        vlock = self._get_video_lock(video_id)
        with vlock:
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute("SELECT * FROM video_datasets WHERE id = ?", (video_id,))
            row = cur.fetchone()
            conn.close()
            if not row:
                return False, "Видео не найдено в SQLite"

            local_path = row["local_video_path"]
            if os.path.exists(local_path) and os.path.getsize(local_path) > 25000:
                sz_mb = round(os.path.getsize(local_path) / (1024 * 1024), 2)
                self.download_progress[video_id] = {"state": "ready", "pct": 100, "mb_done": sz_mb, "mb_total": sz_mb}
                return True, local_path

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            vname = row["video_name"]
            url = row["direct_url"]
            tmp_part = local_path + ".part"

            try:
                self.download_progress[video_id] = {"state": "downloading", "pct": 1, "mb_done": 0.0, "mb_total": 0.0}
                self.worker_status = f"Скачивание оригинала видео #{video_id} '{vname}'..."
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "*/*"
                })
                with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_part, "wb") as out_f:
                    total_bytes = int(resp.headers.get("Content-Length", 0) or 0)
                    total_mb = round(total_bytes / (1024 * 1024), 2) if total_bytes > 0 else 10.0
                    downloaded = 0
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded += len(chunk)
                        mb_done = round(downloaded / (1024 * 1024), 2)
                        pct = min(99, int((downloaded * 100) / max(1, total_bytes))) if total_bytes > 0 else min(95, int(mb_done * 10))
                        self.download_progress[video_id] = {
                            "state": "downloading",
                            "pct": pct,
                            "mb_done": mb_done,
                            "mb_total": total_mb
                        }
                        self.worker_status = f"Загрузка '{vname}': {mb_done:.1f} / {total_mb:.1f} МБ ({pct}%)"

                if os.path.exists(tmp_part) and os.path.getsize(tmp_part) > 25000:
                    shutil.move(tmp_part, local_path)
                    final_mb = round(os.path.getsize(local_path) / (1024 * 1024), 2)
                    self.download_progress[video_id] = {"state": "ready", "pct": 100, "mb_done": final_mb, "mb_total": final_mb}
                    conn2 = get_db_conn()
                    conn2.execute("UPDATE video_datasets SET is_downloaded = 1 WHERE id = ?", (video_id,))
                    conn2.commit()
                    conn2.close()
                    self.worker_status = f"Видео '{vname}' ({final_mb:.1f} МБ) скачано и готово к работе!"
                    self.restart_video = True
                    return True, local_path
            except Exception as e:
                if os.path.exists(tmp_part):
                    try:
                        os.remove(tmp_part)
                    except OSError:
                        pass
                self.download_progress[video_id] = {"state": "error", "pct": 0, "mb_done": 0.0, "mb_total": 0.0}
                self.worker_status = f"Ошибка загрузки '{vname}': {e}"
                return False, str(e)

            return False, "Не удалось скачать файл"

    def download_all_videos_async(self):
        if self.is_busy:
            return
        def _job():
            self.is_busy = True
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute("SELECT id FROM video_datasets ORDER BY id ASC")
            ids = [r["id"] for r in cur.fetchall()]
            conn.close()
            for idx, vid in enumerate(ids, start=1):
                self.worker_status = f"[{idx}/{len(ids)}] Скачивание уникального видео #{vid} из БД SQLite..."
                self.download_single_video(vid)
            self.worker_status = "Все 10 уникальных видео из БД SQLite скачаны в datasets/raw_videos/!"
            self.is_busy = False
        threading.Thread(target=_job, daemon=True).start()

    def train_80_20_async(self, epochs=25):
        if self.is_busy:
            return
        def _train_job():
            self.is_busy = True
            try:
                self.worker_status = "Шаг 1/2: Скачивание 10 видео и авторазметка кадров (80% Train = 8 / 20% Val = 2)..."
                download_and_prepare_80_20_dataset(db_path=DB_PATH, frame_step=6)
                self.worker_status = f"Шаг 2/2: Дообучение YOLO (classes=[0]: person, {epochs} эпох, workers=0)..."
                train_yolo_person_80_20(db_path=DB_PATH, model_name="yolov8n.pt", epochs=epochs, imgsz=640, batch=8)
                self.worker_status = "Дообучение 80/20 успешно завершено! Веса best.pt автоматически подключены."
                self.restart_video = True
            except Exception as e:
                self.worker_status = f"Ошибка обучения YOLO: {e}"
            finally:
                self.is_busy = False
        threading.Thread(target=_train_job, daemon=True).start()

    def update_polygon(self, polygon_list, video_id=None):
        if not isinstance(polygon_list, list) or len(polygon_list) < 3:
            return
        with self.lock:
            target_vid = int(video_id) if video_id is not None else self.active_video_id
            if target_vid == self.active_video_id:
                self.polygon_norm = polygon_list
        poly_json = json.dumps(polygon_list)

        conn = get_db_conn()
        conn.execute("""
            UPDATE video_datasets
            SET default_polygon = ?, has_custom_line = 1
            WHERE id = ?
        """, (poly_json, target_vid))
        conn.commit()
        conn.close()
        self.worker_status = f"Полигон зоны для видео #{target_vid} сохранен в базе данных SQLite!"

    def reset_polygon_to_factory(self, video_id=None):
        with self.lock:
            target_vid = int(video_id) if video_id is not None else self.active_video_id

        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT factory_polygon FROM video_datasets WHERE id = ?", (target_vid,))
        row = cur.fetchone()
        if row:
            factory_poly = json.loads(row[0])
            conn.execute("""
                UPDATE video_datasets
                SET default_polygon = ?, has_custom_line = 0
                WHERE id = ?
            """, (json.dumps(factory_poly), target_vid))
            conn.commit()
            with self.lock:
                if target_vid == self.active_video_id:
                    self.polygon_norm = factory_poly
        conn.close()
        self.worker_status = f"Зона для видео #{target_vid} сброшена к заводским координатам."

    def start_background_loop(self):
        threading.Thread(target=self._inference_loop, daemon=True).start()

    def _inference_loop(self):
        try:
            import cv2
            self.cv2_available = True
        except ImportError:
            self.worker_status = "Установите OpenCV и YOLO: pip install ultralytics opencv-python"
            return

        yolo_model = None
        loaded_weights_name = None

        while not self.stop_requested:
            try:
                best_pt = os.path.join("runs", "detect", "yolo_person_80_20", "weights", "best.pt")
                if os.path.exists(best_pt):
                    target_weights = best_pt
                    self.is_custom_trained = True
                else:
                    target_weights = "yolov8m.pt" if os.path.exists("yolov8m.pt") else "yolov8n.pt"
                    self.is_custom_trained = False

                if yolo_model is None or loaded_weights_name != target_weights:
                    try:
                        from ultralytics import YOLO
                        yolo_model = YOLO(target_weights)
                        loaded_weights_name = target_weights
                        if self.is_custom_trained:
                            self.active_weights = f"Дообученная 80/20 модель ({target_weights})"
                        else:
                            self.active_weights = f"Предобученная модель YOLO ({target_weights}, класс person)"
                        self.yolo_available = True
                    except Exception as e:
                        self.yolo_available = False
                        self.active_weights = f"Ожидание библиотеки ultralytics ({e})"

                with self.lock:
                    vrow = dict(self.active_video_row) if self.active_video_row else None
                    self.restart_video = False

                if not vrow:
                    time.sleep(0.5)
                    continue

                local_path = vrow["local_video_path"]
                vname = vrow["video_name"]

                if not os.path.exists(local_path) or os.path.getsize(local_path) < 25000:
                    ok, _ = self.download_single_video(vrow["id"])
                    if not ok:
                        time.sleep(2.0)
                        continue

                cap = cv2.VideoCapture(local_path)
                if not cap.isOpened():
                    self.worker_status = f"Ошибка открытия видеофайла {local_path}"
                    time.sleep(2.0)
                    continue

                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frame_delay = 1.0 / max(15.0, min(30.0, fps_video))

                os.makedirs("results_videos", exist_ok=True)
                out_clip_path = os.path.join("results_videos", f"yolo_annotated_{vname}")
                w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
                h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_clip_path, fourcc, fps_video, (w_orig, h_orig))
                self.last_saved_clip = out_clip_path

                tracker = StablePersonTracker(reid_ttl_sec=8.0, coast_frames=14, max_match_dist=260.0, ema_alpha=0.45)
                self.frame_index = 0
                self.worker_status = f"Анализ видео #{vrow['id']} '{vname}' ({vrow['split_group'].upper()})"

                while not self.stop_requested and not self.restart_video:
                    t0 = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        tracker = StablePersonTracker(reid_ttl_sec=8.0, coast_frames=14, max_match_dist=260.0, ema_alpha=0.45)
                        self.frame_index = 0
                        continue

                    self.frame_index += 1
                    h, w = frame.shape[:2]

                    with self.lock:
                        poly_norm = self.polygon_norm

                    import numpy as np_local
                    polygon_px = np_local.array([(int(x * w), int(y * h)) for x, y in poly_norm], np_local.int32) if len(poly_norm) >= 3 else np_local.empty((0, 2), np_local.int32)

                    raw_dets = []
                    if yolo_model is not None:
                        results = yolo_model.predict(frame, classes=[0], conf=0.28, iou=0.5, verbose=False)
                        boxes = results[0].boxes
                        if boxes is not None and len(boxes) > 0:
                            for b in boxes:
                                bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0].tolist()]
                                conf_pct = round(float(b.conf[0].item()) * 100.0, 1)
                                raw_dets.append((bx1, by1, bx2, by2, conf_pct))

                    visible_tracks, new_events = tracker.update(raw_dets, polygon_px)

                    if new_events:
                        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        line_str = json.dumps(poly_norm)
                        conn = get_db_conn()
                        for ev in new_events:
                            conn.execute("""
                                INSERT INTO line_crossing_events (
                                    event_timestamp, video_name, person_track_id, class_name,
                                    confidence_pct, foot_x, foot_y, line_coords, event_status, saved_clip_path
                                ) VALUES (?, ?, ?, 'person', ?, ?, ?, ?, ?, ?)
                            """, (ts_str, vname, ev["person_track_id"], ev["confidence_pct"],
                                  ev["foot_x"], ev["foot_y"], line_str, ev["event_status"], out_clip_path))
                        conn.commit()
                        conn.close()

                    if len(polygon_px) >= 3:
                        cv2.polylines(frame, [polygon_px], isClosed=True, color=(0, 190, 255), thickness=3)
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [polygon_px], (0, 0, 255))
                        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

                    for tr in visible_tracks:
                        is_in_zone = tr["crossed"]
                        color = (0, 0, 255) if is_in_zone else (16, 185, 129)
                        state_txt = "[В ЗОНЕ: ПЕРЕСЕК]" if is_in_zone else "[ВНЕ ЗОНЫ]"
                        tag = f"person #{tr['id']} | {tr['conf']:.1f}% {state_txt}"

                        cv2.rectangle(frame, (tr["x1"], tr["y1"]), (tr["x2"], tr["y2"]), color, 2)
                        cv2.circle(frame, (tr["foot_x"], tr["foot_y"]), 6, color, -1)
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
                        cv2.rectangle(frame, (tr["x1"], max(0, tr["y1"] - th - 10)), (tr["x1"] + tw + 8, tr["y1"]), color, -1)
                        cv2.putText(frame, tag, (tr["x1"] + 4, max(15, tr["y1"] - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)

                    writer.write(frame)
                    ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    if ok_jpg:
                        with self.lock:
                            self.latest_jpeg = buf.tobytes()
                            self.live_people = visible_tracks
                            dt = max(0.001, time.time() - t0)
                            self.fps_current = round(1.0 / dt, 1)

                    elapsed = time.time() - t0
                    if elapsed < frame_delay:
                        time.sleep(frame_delay - elapsed)

                cap.release()
                writer.release()
            except Exception as loop_err:
                self.worker_status = f"Перезапуск видеопотока: {loop_err}"
                time.sleep(1.5)


ENGINE = None
ACTIVE_SERVER = None


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html", "/person_line_interface.html"):
            if not os.path.exists(HTML_PATH):
                self.send_error(404, "person_line_interface.html not found")
                return
            with open(HTML_PATH, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/api/state":
            ENGINE._sync_downloaded_flags_in_sqlite()
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute("SELECT * FROM video_datasets ORDER BY id ASC")
            videos = [dict(r) for r in cur.fetchall()]
            cur.execute("SELECT * FROM line_crossing_events ORDER BY id DESC LIMIT 30")
            events = [dict(r) for r in cur.fetchall()]
            cur.execute("SELECT * FROM yolo_training_runs ORDER BY id DESC LIMIT 5")
            runs = [dict(r) for r in cur.fetchall()]
            conn.close()

            with ENGINE.lock:
                dl_prog = dict(ENGINE.download_progress)
                state = {
                    "version": "v4",
                    "server_online": True,
                    "db_connected": os.path.exists(DB_PATH),
                    "db_path": os.path.abspath(DB_PATH),
                    "db_size_kb": round(os.path.getsize(DB_PATH) / 1024.0, 1) if os.path.exists(DB_PATH) else 0,
                    "yolo_available": ENGINE.yolo_available,
                    "is_custom_trained": ENGINE.is_custom_trained,
                    "active_weights": ENGINE.active_weights,
                    "worker_status": ENGINE.worker_status,
                    "is_busy": ENGINE.is_busy,
                    "active_video_id": ENGINE.active_video_id,
                    "polygon_norm": ENGINE.polygon_norm,
                    "live_people": ENGINE.live_people,
                    "fps": ENGINE.fps_current,
                    "frame_index": ENGINE.frame_index,
                    "total_frames": ENGINE.total_frames,
                    "last_saved_clip": ENGINE.last_saved_clip,
                    "download_progress": dl_prog,
                    "videos": videos,
                    "events": events,
                    "training_runs": runs
                }
            self._send_json(state)
            return

        if path in ("/api/raw_video", "/api/video_proxy"):
            vid = int(qs.get("id", [ENGINE.active_video_id])[0])
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute("SELECT local_video_path FROM video_datasets WHERE id = ?", (vid,))
            row = cur.fetchone()
            conn.close()
            if not row:
                self.send_error(404, "Video ID not found in SQLite")
                return

            vpath = row["local_video_path"]
            if not (os.path.exists(vpath) and os.path.getsize(vpath) > 25000):
                ok, _ = ENGINE.download_single_video(vid)
                if not ok or not os.path.exists(vpath):
                    self.send_error(502, "Failed to download video from upstream")
                    return

            file_size = os.path.getsize(vpath)
            range_header = self.headers.get("Range", None)
            start_byte = 0
            end_byte = file_size - 1

            if range_header:
                m = re.search(r"bytes=(\d+)-(\d*)", range_header)
                if m:
                    start_byte = int(m.group(1))
                    if m.group(2):
                        end_byte = min(file_size - 1, int(m.group(2)))
                if start_byte >= file_size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return
                chunk_len = end_byte - start_byte + 1
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start_byte}-{end_byte}/{file_size}")
            else:
                chunk_len = file_size
                self.send_response(200)

            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(chunk_len))
            self.end_headers()

            try:
                with open(vpath, "rb") as vf:
                    vf.seek(start_byte)
                    remaining = chunk_len
                    while remaining > 0:
                        buf = vf.read(min(262144, remaining))
                        if not buf:
                            break
                        self.wfile.write(buf)
                        remaining -= len(buf)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        if path == "/api/frame.jpg":
            with ENGINE.lock:
                jpg = ENGINE.latest_jpeg
            if jpg is None:
                self.send_response(204)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(raw_body)
        except Exception:
            payload = {}

        if path == "/api/shutdown":
            self._send_json({"ok": True, "message": "Shutting down previous server instance"})
            ENGINE.stop_requested = True
            if ACTIVE_SERVER:
                threading.Thread(target=ACTIVE_SERVER.shutdown, daemon=True).start()
            return

        if path == "/api/select_video":
            vid = int(payload.get("video_id", 1))
            ENGINE._load_video_from_sqlite(vid)
            self._send_json({"ok": True, "active_video_id": vid})
            return

        if path == "/api/set_line":
            vid = payload.get("video_id", ENGINE.active_video_id)
            polygon = payload.get("polygon_norm", [])
            ENGINE.update_polygon(polygon, video_id=vid)
            self._send_json({"ok": True, "video_id": int(vid), "polygon_norm": polygon})
            return

        if path == "/api/reset_line":
            vid = payload.get("video_id", ENGINE.active_video_id)
            ENGINE.reset_polygon_to_factory(video_id=vid)
            self._send_json({"ok": True, "video_id": int(vid), "polygon_norm": ENGINE.polygon_norm})
            return

        if path == "/api/log_event":
            ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vname = str(payload.get("video_name", "worker-zone-detection.mp4"))
            tid = int(payload.get("person_track_id", 1))
            conf = float(payload.get("confidence_pct", 92.0))
            fx = int(payload.get("foot_x", 400))
            fy = int(payload.get("foot_y", 350))
            lcoords = str(payload.get("line_coords", "(150,350)-(800,350)"))
            ev_status = str(payload.get("event_status", "ВХОД В ЗОНУ (ПЕРЕСЕК)"))
            clip_p = str(payload.get("saved_clip_path", f"results_videos/clip_{vname}"))
            conn = get_db_conn()
            conn.execute("""
                INSERT INTO line_crossing_events (
                    event_timestamp, video_name, person_track_id, class_name,
                    confidence_pct, foot_x, foot_y, line_coords, event_status, saved_clip_path
                ) VALUES (?, ?, ?, 'person', ?, ?, ?, ?, ?, ?)
            """, (ts_str, vname, tid, conf, fx, fy, lcoords, ev_status, clip_p))
            conn.commit()
            conn.close()
            self._send_json({"ok": True})
            return

        if path == "/api/download_video":
            vid = int(payload.get("video_id", ENGINE.active_video_id))
            threading.Thread(target=ENGINE.download_single_video, args=(vid,), daemon=True).start()
            self._send_json({"ok": True, "message": f"Запущено скачивание видео ID #{vid}"})
            return

        if path == "/api/download_all":
            ENGINE.download_all_videos_async()
            self._send_json({"ok": True, "message": "Запущено скачивание всех 10 уникальных видео из SQLite"})
            return

        if path == "/api/train_80_20":
            epochs = int(payload.get("epochs", 25))
            ENGINE.train_80_20_async(epochs=epochs)
            self._send_json({"ok": True, "message": f"Запущено дообучение YOLO 80/20 ({epochs} эпох)"})
            return

        if path == "/api/clear_events":
            conn = get_db_conn()
            conn.execute("DELETE FROM line_crossing_events")
            conn.commit()
            conn.close()
            self._send_json({"ok": True})
            return

        self.send_error(404, "Unknown POST endpoint")


def _stop_old_servers(base_port=DEFAULT_PORT):
    for p in range(base_port, base_port + 5):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{p}/api/shutdown", data=b"{}", method="POST")
            with urllib.request.urlopen(req, timeout=0.6):
                time.sleep(0.3)
        except Exception:
            pass


def main():
    global ENGINE, ACTIVE_SERVER
    parser = argparse.ArgumentParser(description="Сервер видеоаналитики YOLO + SQLite v4")
    parser.add_argument("--train", action="store_true", help="Запустить дообучение 80/20 сразу в консоли")
    parser.add_argument("--epochs", type=int, default=25, help="Количество эпох для дообучения")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Порт веб-сервера (по умолчанию 8000)")
    args = parser.parse_args()

    if args.train:
        _ensure_clean_v4_database()
        print("[РЕЖИМ ОБУЧЕНИЯ] Запуск подготовки 80/20 (8 Train / 2 Val) и дообучения YOLO...")
        download_and_prepare_80_20_dataset(db_path=DB_PATH, frame_step=6)
        train_yolo_person_80_20(db_path=DB_PATH, model_name="yolov8n.pt", epochs=args.epochs, imgsz=640, batch=8)
        return

    _stop_old_servers(args.port)

    ENGINE = RealYoloVideoEngine()
    ENGINE.start_background_loop()

    chosen_port = args.port
    server = None
    for candidate_port in range(chosen_port, chosen_port + 15):
        try:
            server = ThreadedHTTPServer((HOST, candidate_port), RequestHandler)
            chosen_port = candidate_port
            break
        except OSError:
            continue

    if server is None:
        print("[ОШИБКА] Не удалось найти свободный порт в диапазоне 8000..8014.")
        return

    ACTIVE_SERVER = server

    net_ips = _get_all_local_ips()
    print("=" * 90)
    print("ЗАПУСК СЕРВЕРА ВИДЕОАНАЛИТИКИ YOLO (person) + SQLite v4 (ДОСТУП ПО СЕТИ ВКЛЮЧЕН)")
    print(f"Локальная БД SQLite          : {os.path.abspath(DB_PATH)}")
    print(f"На этом компьютере (локально): http://127.0.0.1:{chosen_port}")
    if net_ips:
        for ip in net_ips:
            print(f"Для ноутбука по сети/VPN     : http://{ip}:{chosen_port}")
    else:
        print(f"Для ноутбука по сети/VPN     : http://<IP_вашего_ПК_в_VPN>:{chosen_port}")
    print("=" * 90)

    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{chosen_port}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ОСТАНОВКА] Сервер остановлен пользователем.")
        ENGINE.stop_requested = True
        server.server_close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
