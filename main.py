import cv2
import time
import math
import threading

import mediapipe as mp
from ultralytics import YOLO
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch


# ============================================================
# AEGIS VISION 3.0
# ============================================================

WINDOW = "AEGIS VISION // JARVIS"

GREEN = (0, 255, 100)
CYAN = (255, 255, 0)
BLUE = (255, 150, 0)
WHITE = (240, 240, 240)
ORANGE = (0, 170, 255)
RED = (0, 80, 255)
MAGENTA = (255, 0, 255)
BLACK = (0, 0, 0)


# ============================================================
# STARTUP
# ============================================================

print()
print("==============================================")
print("          AEGIS VISION 3.0")
print("       MULTIMODAL VISION SYSTEM")
print("==============================================")
print("[SYSTEM] Initializing...")


# ============================================================
# YOLO
# ============================================================

print("[AI] Loading YOLO...")

yolo = YOLO("yolo11n.pt")

print("[AI] YOLO ONLINE")


# ============================================================
# OWLv2
# ============================================================

print("[AI] Loading OWLv2...")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[AI] Device: {device}")

owl_processor = Owlv2Processor.from_pretrained(
    "google/owlv2-base-patch16-ensemble"
)

owl_model = Owlv2ForObjectDetection.from_pretrained(
    "google/owlv2-base-patch16-ensemble"
)

owl_model.to(device)
owl_model.eval()

print("[AI] OWLv2 ONLINE")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera unavailable")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW,
    1280,
    720
)

fullscreen = False


# ============================================================
# STATE
# ============================================================

previous_time = time.time()

fps = 0

gesture = "NONE"
previous_gesture = "NONE"

pointer_x = 640
pointer_y = 360

mode = "VISION"

selected_object = None

search_query = ""

search_mode = False

owl_results = []

owl_lock = threading.Lock()

owl_running = False

last_owl_time = 0

last_action_time = 0


# ============================================================
# OBJECT CLASSES
# ============================================================

ANIMALS = {
    "dog",
    "cat",
    "bird",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
}


OBJECT_COLORS = {
    "person": GREEN,
    "dog": ORANGE,
    "cat": ORANGE,
    "bird": CYAN,
    "cell phone": MAGENTA,
    "laptop": CYAN,
    "keyboard": BLUE,
    "mouse": MAGENTA,
    "chair": ORANGE,
    "couch": ORANGE,
    "tv": BLUE,
    "bottle": GREEN,
    "cup": CYAN,
    "book": BLUE,
    "backpack": MAGENTA,
    "handbag": MAGENTA,
    "remote": MAGENTA,
    "clock": CYAN,
    "vase": ORANGE,
    "potted plant": GREEN,
}


# ============================================================
# HELPERS
# ============================================================

def distance(a, b):

    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def draw_text(
    frame,
    text,
    x,
    y,
    scale=0.5,
    color=GREEN,
    thickness=1
):

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def corner_box(
    frame,
    x,
    y,
    w,
    h,
    color=GREEN
):

    if w <= 5 or h <= 5:
        return

    length = max(
        12,
        min(w, h) // 5
    )

    t = 2

    # top left

    cv2.line(
        frame,
        (x, y),
        (x + length, y),
        color,
        t
    )

    cv2.line(
        frame,
        (x, y),
        (x, y + length),
        color,
        t
    )

    # top right

    cv2.line(
        frame,
        (x + w, y),
        (x + w - length, y),
        color,
        t
    )

    cv2.line(
        frame,
        (x + w, y),
        (x + w, y + length),
        color,
        t
    )

    # bottom left

    cv2.line(
        frame,
        (x, y + h),
        (x + length, y + h),
        color,
        t
    )

    cv2.line(
        frame,
        (x, y + h),
        (x, y + h - length),
        color,
        t
    )

    # bottom right

    cv2.line(
        frame,
        (x + w, y + h),
        (x + w - length, y + h),
        color,
        t
    )

    cv2.line(
        frame,
        (x + w, y + h),
        (x + w, y + h - length),
        color,
        t
    )


# ============================================================
# GESTURE
# ============================================================

def detect_gesture(hand):

    if hand is None:

        return "NONE", None

    lm = hand.landmark

    p = [
        (x.x, x.y)
        for x in lm
    ]

    thumb_tip = p[4]
    thumb_ip = p[3]

    index_tip = p[8]
    index_pip = p[6]

    middle_tip = p[12]
    middle_pip = p[10]

    ring_tip = p[16]
    ring_pip = p[14]

    pinky_tip = p[20]
    pinky_pip = p[18]

    # pinch

    if distance(
        thumb_tip,
        index_tip
    ) < 0.055:

        return "PINCH", index_tip

    index_up = (
        index_tip[1] <
        index_pip[1] - 0.035
    )

    middle_up = (
        middle_tip[1] <
        middle_pip[1] - 0.035
    )

    ring_up = (
        ring_tip[1] <
        ring_pip[1] - 0.035
    )

    pinky_up = (
        pinky_tip[1] <
        pinky_pip[1] - 0.035
    )

    # thumbs up

    if (
        thumb_tip[1] <
        thumb_ip[1] - 0.08
        and
        not index_up
        and
        not middle_up
        and
        not ring_up
        and
        not pinky_up
    ):

        return "THUMBS UP", thumb_tip

    # V

    if (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
    ):

        return "V SIGN", index_tip

    # point

    if (
        index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    ):

        return "POINT", index_tip

    # open hand

    if (
        index_up
        and middle_up
        and ring_up
        and pinky_up
    ):

        return "OPEN HAND", index_tip

    # fist

    if (
        not index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    ):

        return "FIST", p[0]

    return "NONE", index_tip


# ============================================================
# OWLv2 SEARCH
# ============================================================

def owl_search(frame, query):

    global owl_running
    global owl_results

    if not query.strip():
        return

    if owl_running:
        return

    owl_running = True

    try:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        texts = [
            query.strip()
        ]

        inputs = owl_processor(
            text=texts,
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = owl_model(
                **inputs
            )

        target_sizes = torch.tensor(
            [image.size[::-1]]
        ).to(device)

        results = owl_processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.15,
            target_sizes=target_sizes
        )[0]

        new_results = []

        for score, box in zip(
            results["scores"],
            results["boxes"]
        ):

            score = float(score)

            box = [
                int(v)
                for v in box.cpu().tolist()
            ]

            new_results.append(
                (
                    query,
                    score,
                    box
                )
            )

        with owl_lock:

            owl_results = new_results

    except Exception as e:

        print(
            "[OWLV2 ERROR]",
            e
        )

    owl_running = False


# ============================================================
# DRAW OWLv2
# ============================================================

def draw_owl_results(frame):

    with owl_lock:

        results = list(
            owl_results
        )

    for i, result in enumerate(results):

        name, confidence, box = result

        x1, y1, x2, y2 = box

        corner_box(
            frame,
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            MAGENTA
        )

        draw_text(
            frame,
            f"AI SEARCH // {name.upper()} "
            f"{confidence * 100:.0f}%",
            x1,
            max(25, y1 - 8),
            0.45,
            MAGENTA,
            1
        )


# ============================================================
# YOLO
# ============================================================

def run_yolo(frame):

    objects = []

    results = yolo(
        frame,
        imgsz=640,
        conf=0.35,
        verbose=False
    )

    number = 1

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            name = yolo.names[cls]

            coords = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = map(
                int,
                coords
            )

            objects.append(
                (
                    name,
                    confidence,
                    (x1, y1, x2, y2),
                    number
                )
            )

            number += 1

    return objects


# ============================================================
# OBJECT UNDER POINTER
# ============================================================

def object_under_pointer(
    objects,
    px,
    py
):

    for obj in objects:

        name, confidence, box, number = obj

        x1, y1, x2, y2 = box

        if (
            x1 <= px <= x2
            and
            y1 <= py <= y2
        ):

            return obj

    return None


# ============================================================
# MAIN
# ============================================================

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(
            frame,
            1
        )

        height, width = frame.shape[:2]

        # ====================================================
        # FPS
        # ====================================================

        now = time.time()

        delta = now - previous_time

        if delta > 0:

            instant_fps = 1 / delta

            fps = (
                fps * 0.9 +
                instant_fps * 0.1
            )

        previous_time = now

        # ====================================================
        # MEDIAPIPE
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        mp_results = holistic.process(
            rgb
        )

        rgb.flags.writeable = True

        image = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )

        # ====================================================
        # FACE
        # ====================================================

        face = (
            mp_results.face_landmarks
            is not None
        )

        if face:

            mp_drawing.draw_landmarks(
                image,
                mp_results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(
                    color=CYAN,
                    thickness=1,
                    circle_radius=1
                ),
                mp_drawing.DrawingSpec(
                    color=BLUE,
                    thickness=1
                )
            )

        # ====================================================
        # HANDS
        # ====================================================

        active_hand = None

        active_point = None

        right_gesture = "NONE"

        left_gesture = "NONE"

        if mp_results.right_hand_landmarks:

            right_gesture, right_point = detect_gesture(
                mp_results.right_hand_landmarks
            )

            active_hand = "RIGHT"

            if right_point:

                active_point = (
                    right_point[0] * width,
                    right_point[1] * height
                )

            mp_drawing.draw_landmarks(
                image,
                mp_results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=GREEN,
                    thickness=2,
                    circle_radius=2
                ),
                mp_drawing.DrawingSpec(
                    color=CYAN,
                    thickness=2
                )
            )

        if mp_results.left_hand_landmarks:

            left_gesture, left_point = detect_gesture(
                mp_results.left_hand_landmarks
            )

            if active_point is None:

                active_hand = "LEFT"

                if left_point:

                    active_point = (
                        left_point[0] * width,
                        left_point[1] * height
                    )

            mp_drawing.draw_landmarks(
                image,
                mp_results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=GREEN,
                    thickness=2,
                    circle_radius=2
                ),
                mp_drawing.DrawingSpec(
                    color=CYAN,
                    thickness=2
                )
            )

        # ====================================================
        # ACTIVE GESTURE
        # ====================================================

        if active_hand == "RIGHT":

            gesture = right_gesture

        elif active_hand == "LEFT":

            gesture = left_gesture

        else:

            gesture = "NONE"

        # ====================================================
        # POINTER
        # ====================================================

        if active_point:

            target_x = active_point[0]
            target_y = active_point[1]

            pointer_x = (
                pointer_x * 0.8 +
                target_x * 0.2
            )

            pointer_y = (
                pointer_y * 0.8 +
                target_y * 0.2
            )

        # ====================================================
        # YOLO
        # ====================================================

        objects = run_yolo(
            image
        )

        # ====================================================
        # OBJECT COUNTS
        # ====================================================

        people = sum(
            1
            for obj in objects
            if obj[0] == "person"
        )

        animals = sum(
            1
            for obj in objects
            if obj[0] in ANIMALS
        )

        # ====================================================
        # DRAW YOLO OBJECTS
        # ====================================================

        for obj in objects:

            name, confidence, box, number = obj

            x1, y1, x2, y2 = box

            color = OBJECT_COLORS.get(
                name,
                GREEN
            )

            selected = (
                selected_object == obj
            )

            if selected:

                color = MAGENTA

            corner_box(
                image,
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                color
            )

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.circle(
                image,
                (cx, cy),
                4,
                color,
                -1
            )

            label = (
                f"{number:02d} "
                f"{name.upper()} "
                f"{confidence * 100:.0f}%"
            )

            draw_text(
                image,
                label,
                x1,
                max(20, y1 - 8),
                0.42,
                color
            )

        # ====================================================
        # OWLv2
        # ====================================================

        draw_owl_results(
            image
        )

        # ====================================================
        # GESTURE ACTIONS
        # ====================================================

        if (
            gesture == "V SIGN"
            and
            previous_gesture != "V SIGN"
            and
            time.time() - last_action_time > 1
        ):

            modes = [
                "VISION",
                "GESTURE",
                "OBJECT SCAN",
                "TARGET"
            ]

            index = modes.index(
                mode
            )

            index += 1

            if index >= len(modes):

                index = 0

            mode = modes[index]

            print(
                "[GESTURE] MODE:",
                mode
            )

            last_action_time = time.time()

        if (
            gesture == "PINCH"
            and
            previous_gesture != "PINCH"
        ):

            target = object_under_pointer(
                objects,
                int(pointer_x),
                int(pointer_y)
            )

            if target:

                selected_object = target

                print(
                    "[TARGET LOCK]",
                    target[0]
                )

        if (
            gesture == "OPEN HAND"
            and
            previous_gesture != "OPEN HAND"
        ):

            selected_object = None

            print(
                "[TARGET] RELEASE"
            )

        if (
            gesture == "THUMBS UP"
            and
            previous_gesture != "THUMBS UP"
        ):

            if selected_object:

                print(
                    "[TARGET CONFIRMED]",
                    selected_object[0]
                )

        previous_gesture = gesture

        # ====================================================
        # START OWLv2 SEARCH
        # ====================================================

        if (
            search_query
            and
            not owl_running
            and
            time.time() - last_owl_time > 1.5
        ):

            search_frame = image.copy()

            thread = threading.Thread(
                target=owl_search,
                args=(
                    search_frame,
                    search_query
                ),
                daemon=True
            )

            thread.start()

            last_owl_time = time.time()

        # ====================================================
        # POINTER
        # ====================================================

        if active_point:

            px = int(pointer_x)
            py = int(pointer_y)

            pc = (
                MAGENTA
                if gesture == "PINCH"
                else GREEN
            )

            cv2.circle(
                image,
                (px, py),
                24,
                pc,
                1
            )

            cv2.circle(
                image,
                (px, py),
                4,
                pc,
                -1
            )

            cv2.line(
                image,
                (px - 38, py),
                (px - 8, py),
                pc,
                1
            )

            cv2.line(
                image,
                (px + 8, py),
                (px + 38, py),
                pc,
                1
            )

            cv2.line(
                image,
                (px, py - 38),
                (px, py - 8),
                pc,
                1
            )

            cv2.line(
                image,
                (px, py + 8),
                (px, py + 38),
                pc,
                1
            )

        # ====================================================
        # HUD
        # ====================================================

        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (width, 105),
            BLACK,
            -1
        )

        cv2.rectangle(
            overlay,
            (0, height - 105),
            (width, height),
            BLACK,
            -1
        )

        image = cv2.addWeighted(
            overlay,
            0.72,
            image,
            0.28,
            0
        )

        # HEADER

        draw_text(
            image,
            "AEGIS",
            25,
            38,
            0.95,
            GREEN,
            2
        )

        draw_text(
            image,
            "MULTIMODAL VISION CORE",
            25,
            67,
            0.42,
            CYAN
        )

        draw_text(
            image,
            "ONLINE",
            width - 125,
            32,
            0.42,
            GREEN
        )

        draw_text(
            image,
            f"{fps:.1f} FPS",
            width - 125,
            58,
            0.42,
            CYAN
        )

        draw_text(
            image,
            f"MODE // {mode}",
            width - 180,
            82,
            0.38,
            WHITE
        )

        # LEFT INFO

        draw_text(
            image,
            "BIOMETRIC",
            25,
            135,
            0.43,
            CYAN
        )

        draw_text(
            image,
            f"FACE     {'LOCKED' if face else 'SEARCH'}",
            25,
            158,
            0.38
        )

        draw_text(
            image,
            f"R-HAND   {right_gesture}",
            25,
            180,
            0.38
        )

        draw_text(
            image,
            f"L-HAND   {left_gesture}",
            25,
            202,
            0.38
        )

        # RIGHT INFO

        rx = width - 220

        draw_text(
            image,
            "ENVIRONMENT",
            rx,
            135,
            0.43,
            CYAN
        )

        draw_text(
            image,
            f"PEOPLE    {people}",
            rx,
            158,
            0.38
        )

        draw_text(
            image,
            f"ANIMALS   {animals}",
            rx,
            180,
            0.38
        )

        draw_text(
            image,
            f"OBJECTS   {len(objects)}",
            rx,
            202,
            0.38
        )

        # SEARCH PANEL

        if search_mode:

            panel_w = 550
            panel_h = 80

            panel_x = (
                width // 2 -
                panel_w // 2
            )

            panel_y = 115

            cv2.rectangle(
                image,
                (
                    panel_x,
                    panel_y
                ),
                (
                    panel_x + panel_w,
                    panel_y + panel_h
                ),
                BLACK,
                -1
            )

            corner_box(
                image,
                panel_x,
                panel_y,
                panel_w,
                panel_h,
                MAGENTA
            )

            draw_text(
                image,
                "OPEN VOCABULARY SEARCH",
                panel_x + 15,
                panel_y + 28,
                0.42,
                MAGENTA
            )

            draw_text(
                image,
                "QUERY: " + search_query + "_",
                panel_x + 15,
                panel_y + 58,
                0.5,
                WHITE
            )

        # TARGET PANEL

        if selected_object:

            name = selected_object[0]

            panel_x = (
                width // 2 - 180
            )

            panel_y = height - 175

            cv2.rectangle(
                image,
                (
                    panel_x,
                    panel_y
                ),
                (
                    panel_x + 360,
                    panel_y + 60
                ),
                BLACK,
                -1
            )

            corner_box(
                image,
                panel_x,
                panel_y,
                360,
                60,
                MAGENTA
            )

            draw_text(
                image,
                "TARGET LOCK",
                panel_x + 15,
                panel_y + 24,
                0.4,
                MAGENTA
            )

            draw_text(
                image,
                name.upper(),
                panel_x + 15,
                panel_y + 48,
                0.5,
                WHITE
            )

        # FOOTER

        draw_text(
            image,
            "[T] SEARCH",
            25,
            height - 65,
            0.38
        )

        draw_text(
            image,
            "[F] FULLSCREEN",
            140,
            height - 65,
            0.38
        )

        draw_text(
            image,
            "[SPACE] CAPTURE",
            285,
            height - 65,
            0.38
        )

        draw_text(
            image,
            "[Q] EXIT",
            455,
            height - 65,
            0.38
        )

        draw_text(
            image,
            "GESTURE // " + gesture,
            width - 260,
            height - 65,
            0.38,
            MAGENTA if gesture == "PINCH" else GREEN
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            WINDOW,
            image
        )

        # ====================================================
        # INPUT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            27
        ):

            break

        # fullscreen

        if key == ord("f"):

            fullscreen = not fullscreen

            if fullscreen:

                cv2.setWindowProperty(
                    WINDOW,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                )

            else:

                cv2.setWindowProperty(
                    WINDOW,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL
                )

                cv2.resizeWindow(
                    WINDOW,
                    1280,
                    720
                )

        # screenshot

        if key == 32:

            filename = (
                f"aegis_"
                f"{int(time.time())}.png"
            )

            cv2.imwrite(
                filename,
                image
            )

            print(
                "[CAPTURE]",
                filename
            )

        # search mode

        if key == ord("t"):

            search_mode = True

            search_query = ""

            print(
                "[SEARCH] Type object and press ENTER"
            )

        # typing

        if search_mode:

            if key == 13:

                search_mode = False

                owl_results = []

                last_owl_time = 0

                print(
                    "[SEARCH]",
                    search_query
                )

            elif key == 8:

                search_query = search_query[:-1]

            elif (
                32 <= key <= 126
            ):

                search_query += chr(key)


# ============================================================
# SHUTDOWN
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("[SYSTEM] AEGIS OFFLINE")