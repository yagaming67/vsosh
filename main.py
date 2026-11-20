import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Укажите путь к вашей фотографии поля
IMAGE_PATH = "field3.jpg"

# Диапазоны цветов в пространстве HSV (используем кортежи, чтобы не терялись при копировании)
COLOR_RANGES = {
    "green":  ((np.array((35, 60, 60)),  np.array((85, 255, 255))),),
    "blue":   ((np.array((95, 70, 60)),  np.array((130, 255, 255))),),
    "yellow": ((np.array((18, 80, 80)),  np.array((34, 255, 255))),),
    "purple": ((np.array((131, 60, 60)), np.array((165, 255, 255))),),
    "red":    (
        (np.array((0, 70, 60)),   np.array((10, 255, 255))),
        (np.array((170, 70, 60)), np.array((180, 255, 255)))
    )
}

ROWS = ("А", "Б", "В", "Г", "Д", "Е")
FIELD_MARKERS = (27, 28, 29)

def mm_to_cell(x_mm, y_mm):
    """Перевод координат в миллиметрах в название клетки (например, Б3)"""
    col = round(x_mm / 28.0)
    row = round(y_mm / 28.0)
    
    if 1 <= row <= 6:
        row_name = ROWS[row - 1]
        if 0 <= col <= 8:
            return f"{row_name}{col}"
        elif col == 9 and 1 <= row <= 5:
            return f"Отбой-{row_name}"
    return "Вне сетки"

def analyze_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Ошибка: не удалось открыть файл '{image_path}'")
        return

    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. ПОИСК ВСЕХ МАРКЕРОВ ARUCO (DICT_4X4_50 из регламента)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    pixel_markers = {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(output, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i].reshape(4, 2)
            center = pts.mean(axis=0)
            pixel_markers[int(marker_id)] = center

    # 2. ПРИВЯЗКА К БАЗОВЫМ МЕТКАМ ПОЛЯ (ID 27, 28, 29)
    M = None
    if 27 in pixel_markers and 28 in pixel_markers and 29 in pixel_markers:
        src_pts = np.float32((
            pixel_markers.get(27), # (0, 0) мм
            pixel_markers.get(28), # (224, 0) мм
            pixel_markers.get(29)  # (252, 168) мм
        ))
        dst_pts = np.float32((
            (0.0, 0.0),
            (224.0, 0.0),
            (252.0, 168.0)
        ))
        M = cv2.getAffineTransform(src_pts, dst_pts)
        print("Базовые метки поля (27, 28, 29) найдены! Координаты считаются в мм.\n")
    else:
        print("Внимание: не все базовые метки поля (27, 28, 29) видны на фото. Координаты будут в пикселях.\n")

    def get_coords_text(px, py):
        if M is not None:
            pt = np.array((((px, py),),), dtype=np.float32)
            mm = cv2.transform(pt, M).flatten()
            x_val, y_val = float(mm.item(0)), float(mm.item(1))
            cell = mm_to_cell(x_val, y_val)
            return f"X={x_val:.1f}мм, Y={y_val:.1f}мм [{cell}]"
        return f"px=({int(px)}, {int(py)})"

    # 3. ВЫВОД НАЙДЕННЫХ ARUCO-ШАШЕК (кроме меток поля 27, 28, 29)
    print("--- МАРКЕРЫ ARUCO НА ШАШКАХ ---")
    for marker_id, center in pixel_markers.items():
        if marker_id not in FIELD_MARKERS:
            cx, cy = float(center.item(0)), float(center.item(1))
            info = get_coords_text(cx, cy)
            print(f"ArUco ID {marker_id}: {info}")
            cv2.putText(output, f"ID:{marker_id}", (int(cx)-20, int(cy)-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 4. ПОИСК QR-КОДОВ НА ШАШКАХ
    print("\n--- QR-КОДЫ НА ШАШКАХ ---")
    qr_codes = decode(gray)
    for qr in qr_codes:
        text = qr.data.decode("utf-8")
        pts = np.array(tuple((p.x, p.y) for p in qr.polygon), np.int32)
        cv2.polylines(output, (pts,), True, (255, 165, 0), 2)
        cx = int(np.mean(tuple(p.x for p in qr.polygon)))
        cy = int(np.mean(tuple(p.y for p in qr.polygon)))
        info = get_coords_text(cx, cy)
        print(f"QR '{text}': {info}")
        cv2.putText(output, f"QR:{text}", (cx-20, cy-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

    # 5. ПОИСК ЦВЕТНЫХ ШАШЕК
    print("\n--- ЦВЕТНЫЕ ШАШКИ ---")
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    min_area, max_area = 300, 15000
    if 27 in pixel_markers and 28 in pixel_markers:
        dist_px = np.linalg.norm(pixel_markers.get(28) - pixel_markers.get(27))
        checker_r_px = (13.0 / 224.0) * dist_px
        expected_area = np.pi * (checker_r_px ** 2)
        min_area = expected_area * 0.4
        max_area = expected_area * 1.8

    for color_name, ranges in COLOR_RANGES.items():
        full_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            full_mask |= cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                if circularity > 0.65:
                    M_cnt = cv2.moments(cnt)
                    cx = int(M_cnt.get("m10") / M_cnt.get("m00"))
                    cy = int(M_cnt.get("m01") / M_cnt.get("m00"))
                    info = get_coords_text(cx, cy)
                    print(f"Цвет [{color_name.upper()}]: {info}")

                    cv2.drawContours(output, (cnt,), -1, (0, 255, 0), 2)
                    cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(output, color_name, (cx-25, cy+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite("result.jpg", output)
    print("\nГотово! Размеченное изображение сохранено в 'result.jpg'.")

if __name__ == "__main__":
    analyze_image(IMAGE_PATH)
