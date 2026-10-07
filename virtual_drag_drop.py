import cv2
import mediapipe as mp
import math
import random

# =========================================================
# SETTINGS
# =========================================================

WIDTH = 1280
HEIGHT = 720

OBJECT_RADIUS = 32
TARGET_W = 130
TARGET_H = 105

PINCH_RATIO = 0.45
GRAB_DISTANCE = 70
DROP_DISTANCE = 85

# OpenCV = BGR
COLORS = {
    "RED": (0, 0, 255),
    "BLUE": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "YELLOW": (0, 255, 255),
    "PURPLE": (255, 0, 255)
}

NAMES = list(COLORS.keys())

# =========================================================
# OBJECTS
# =========================================================

object_x = [140, 390, 640, 890, 1140]

objects = []

for i, name in enumerate(NAMES):
    objects.append({
        "name": name,
        "x": object_x[i],
        "y": 150,
        "done": False
    })

# =========================================================
# TARGET BOXES
# =========================================================

targets = []

for i, name in enumerate(NAMES):
    targets.append({
        "name": name,
        "x": object_x[i],
        "y": 570
    })

# =========================================================
# GAME VARIABLES
# =========================================================

score = 0
selected = None
offset_x = 0
offset_y = 0

# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# =========================================================
# FUNCTIONS
# =========================================================

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def reset_game():

    global score
    global selected

    score = 0
    selected = None

    for i, obj in enumerate(objects):

        obj["x"] = object_x[i]
        obj["y"] = 150
        obj["done"] = False


def random_position():

    return (
        random.randint(100, WIDTH - 100),
        random.randint(120, 250)
    )


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # =====================================================
    # IMPORTANT:
    # PROCESS RAW CAMERA BEFORE DRAWING ANY GAME GRAPHICS
    # =====================================================

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    finger_x = None
    finger_y = None
    pinching = False

    # =====================================================
    # HAND DETECTION
    # =====================================================

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        thumb = hand.landmark[
            mp_hands.HandLandmark.THUMB_TIP
        ]

        index = hand.landmark[
            mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]

        wrist = hand.landmark[
            mp_hands.HandLandmark.WRIST
        ]

        middle = hand.landmark[
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
        ]

        # Pixel coordinates
        tx = int(thumb.x * WIDTH)
        ty = int(thumb.y * HEIGHT)

        ix = int(index.x * WIDTH)
        iy = int(index.y * HEIGHT)

        wx = int(wrist.x * WIDTH)
        wy = int(wrist.y * HEIGHT)

        mx = int(middle.x * WIDTH)
        my = int(middle.y * HEIGHT)

        # Cursor = midpoint between thumb and index
        finger_x = (tx + ix) // 2
        finger_y = (ty + iy) // 2

        # Pinch distance
        pinch_distance = distance(
            tx, ty,
            ix, iy
        )

        # Hand size
        hand_size = distance(
            wx, wy,
            mx, my
        )

        if hand_size > 0:

            pinch_ratio = pinch_distance / hand_size

            if pinch_ratio < PINCH_RATIO:
                pinching = True

        # =================================================
        # DRAW HAND
        # =================================================

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Cursor
        cv2.circle(
            frame,
            (finger_x, finger_y),
            12,
            (255, 255, 255),
            -1
        )

        cv2.circle(
            frame,
            (finger_x, finger_y),
            15,
            (0, 0, 0),
            2
        )

    # =====================================================
    # GRAB OBJECT
    # =====================================================

    if (
        finger_x is not None
        and pinching
        and selected is None
    ):

        for obj in objects:

            if obj["done"]:
                continue

            d = distance(
                finger_x,
                finger_y,
                obj["x"],
                obj["y"]
            )

            if d < GRAB_DISTANCE:

                selected = obj

                offset_x = obj["x"] - finger_x
                offset_y = obj["y"] - finger_y

                break

    # =====================================================
    # DRAG OBJECT
    # =====================================================

    if (
        selected is not None
        and finger_x is not None
        and pinching
    ):

        selected["x"] = finger_x + offset_x
        selected["y"] = finger_y + offset_y

        # Keep inside camera
        selected["x"] = max(
            OBJECT_RADIUS,
            min(
                WIDTH - OBJECT_RADIUS,
                selected["x"]
            )
        )

        selected["y"] = max(
            100,
            min(
                HEIGHT - OBJECT_RADIUS - 10,
                selected["y"]
            )
        )

    # =====================================================
    # RELEASE / DROP
    # =====================================================

    if not pinching and selected is not None:

        correct = False

        for target in targets:

            d = distance(
                selected["x"],
                selected["y"],
                target["x"],
                target["y"]
            )

            if d < DROP_DISTANCE:

                if selected["name"] == target["name"]:

                    selected["x"] = target["x"]
                    selected["y"] = target["y"]

                    selected["done"] = True

                    score += 1

                    correct = True

                break

        # Wrong target
        if not correct:

            selected["x"], selected["y"] = random_position()

        selected = None

    # =====================================================
    # TOP INSTRUCTION BAR
    # =====================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (WIDTH, 75),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "VIRTUAL DRAG & DROP",
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Pinch to Grab",
        (410, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Drag to Matching Box",
        (620, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"SCORE {score}/5",
        (1080, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # =====================================================
    # TARGET AREA
    # =====================================================

    # Transparent-looking dark target panel
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 430),
        (WIDTH, HEIGHT),
        (25, 25, 25),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0
    )

    # Target heading
    cv2.putText(
        frame,
        "MATCHING TARGETS",
        (520, 465),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =====================================================
    # DRAW TARGET BOXES
    # =====================================================

    for target in targets:

        color = COLORS[target["name"]]

        x = target["x"]
        y = target["y"]

        # Outer colored box
        cv2.rectangle(
            frame,
            (
                x - TARGET_W // 2,
                y - TARGET_H // 2
            ),
            (
                x + TARGET_W // 2,
                y + TARGET_H // 2
            ),
            color,
            4
        )

        # Inner dark box
        cv2.rectangle(
            frame,
            (
                x - TARGET_W // 2 + 8,
                y - TARGET_H // 2 + 8
            ),
            (
                x + TARGET_W // 2 - 8,
                y + TARGET_H // 2 - 8
            ),
            (35, 35, 35),
            -1
        )

        # Target name
        text_size = cv2.getTextSize(
            target["name"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )[0]

        cv2.putText(
            frame,
            target["name"],
            (
                x - text_size[0] // 2,
                y + 7
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # =====================================================
    # DRAW DRAGGABLE OBJECTS
    # =====================================================

    for obj in objects:

        if obj["done"]:
            continue

        color = COLORS[obj["name"]]

        x = obj["x"]
        y = obj["y"]

        # Highlight selected object
        if selected == obj:

            cv2.circle(
                frame,
                (x, y),
                OBJECT_RADIUS + 8,
                (255, 255, 255),
                3
            )

        # Colored circle
        cv2.circle(
            frame,
            (x, y),
            OBJECT_RADIUS,
            color,
            -1
        )

        # White border
        cv2.circle(
            frame,
            (x, y),
            OBJECT_RADIUS,
            (255, 255, 255),
            2
        )

        # Object label
        text_size = cv2.getTextSize(
            obj["name"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )[0]

        cv2.putText(
            frame,
            obj["name"],
            (
                x - text_size[0] // 2,
                y + 55
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # =====================================================
    # STATUS
    # =====================================================

    if pinching:

        status = "GRABBING"

    else:

        status = "READY"

    cv2.putText(
        frame,
        status,
        (30, 690),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "R = Restart     ESC = Exit",
        (960, 690),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        2
    )

    # =====================================================
    # COMPLETION MESSAGE
    # =====================================================

    if score == 5:

        cv2.rectangle(
            frame,
            (390, 280),
            (890, 390),
            (20, 20, 20),
            -1
        )

        cv2.rectangle(
            frame,
            (390, 280),
            (890, 390),
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "GAME COMPLETED!",
            (475, 335),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "Press R to Restart",
            (520, 370),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "Virtual Drag and Drop Game",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key == ord("r") or key == ord("R"):
        reset_game()


# =========================================================
# CLEANUP
# =========================================================

cap.release()
hands.close()
cv2.destroyAllWindows()