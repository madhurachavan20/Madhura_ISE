import cv2
import mediapipe as mp
import random
import math

# ============================================================
# VIRTUAL DRAG AND DROP GAME
# OpenCV + MediaPipe
# ============================================================

# -----------------------------
# Game settings
# -----------------------------
WIDTH = 1280
HEIGHT = 720

BOX_SIZE = 80
TARGET_SIZE = 120

TOTAL_OBJECTS = 5

# Colors in BGR format
COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "YELLOW": (0, 255, 255),
    "PURPLE": (255, 0, 255)
}

COLOR_NAMES = list(COLORS.keys())


# -----------------------------
# Create draggable objects
# -----------------------------
def create_objects():

    objects = []

    positions = [
        (150, 150),
        (350, 180),
        (550, 130),
        (750, 180),
        (950, 130)
    ]

    for i in range(TOTAL_OBJECTS):

        color_name = COLOR_NAMES[i]
        color = COLORS[color_name]

        x, y = positions[i]

        obj = {
            "x": x,
            "y": y,
            "size": BOX_SIZE,
            "color": color,
            "name": color_name,
            "dragging": False,
            "placed": False
        }

        objects.append(obj)

    return objects


# -----------------------------
# Create target areas
# -----------------------------
def create_targets():

    targets = []

    positions = [
        (150, 520),
        (350, 520),
        (550, 520),
        (750, 520),
        (950, 520)
    ]

    for i in range(TOTAL_OBJECTS):

        color_name = COLOR_NAMES[i]
        color = COLORS[color_name]

        x, y = positions[i]

        target = {
            "x": x,
            "y": y,
            "size": TARGET_SIZE,
            "color": color,
            "name": color_name
        }

        targets.append(target)

    return targets


# -----------------------------
# Check if point is inside box
# -----------------------------
def point_inside_box(px, py, x, y, size):

    return (
        x <= px <= x + size and
        y <= py <= y + size
    )


# -----------------------------
# Check distance
# -----------------------------
def distance(x1, y1, x2, y2):

    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )


# -----------------------------
# Check whether object is
# correctly placed
# -----------------------------
def check_placement(obj, target):

    object_center_x = obj["x"] + obj["size"] // 2
    object_center_y = obj["y"] + obj["size"] // 2

    target_center_x = target["x"] + target["size"] // 2
    target_center_y = target["y"] + target["size"] // 2

    d = distance(
        object_center_x,
        object_center_y,
        target_center_x,
        target_center_y
    )

    return (
        d < 60 and
        obj["name"] == target["name"]
    )


# ============================================================
# MediaPipe setup
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ============================================================
# Camera setup
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


if not cap.isOpened():

    print("ERROR: Cannot open webcam.")

    exit()


# ============================================================
# Initialize game
# ============================================================

objects = create_objects()
targets = create_targets()

score = 0
dragged_object = None

game_completed = False


# ============================================================
# Main game loop
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Cannot read webcam.")
        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Detect hand
    results = hands.process(rgb)


    index_x = None
    index_y = None

    # --------------------------------------------------------
    # Hand detected
    # --------------------------------------------------------

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


        # Index finger tip
        index_tip = hand_landmarks.landmark[
            mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]


        index_x = int(
            index_tip.x * WIDTH
        )

        index_y = int(
            index_tip.y * HEIGHT
        )


        # Draw cursor
        cv2.circle(
            frame,
            (index_x, index_y),
            12,
            (255, 255, 255),
            -1
        )

        cv2.circle(
            frame,
            (index_x, index_y),
            16,
            (0, 0, 0),
            2
        )


        # ----------------------------------------------------
        # Thumb + index finger distance
        # ----------------------------------------------------

        thumb_tip = hand_landmarks.landmark[
            mp_hands.HandLandmark.THUMB_TIP
        ]

        thumb_x = int(
            thumb_tip.x * WIDTH
        )

        thumb_y = int(
            thumb_tip.y * HEIGHT
        )


        pinch_distance = distance(
            index_x,
            index_y,
            thumb_x,
            thumb_y
        )


        # Draw line between thumb and index
        cv2.line(
            frame,
            (index_x, index_y),
            (thumb_x, thumb_y),
            (255, 255, 255),
            2
        )


        # Pinch detection
        is_pinching = pinch_distance < 45


        # ----------------------------------------------------
        # START DRAGGING
        # ----------------------------------------------------

        if is_pinching and dragged_object is None:

            for obj in objects:

                if obj["placed"]:
                    continue

                if point_inside_box(
                    index_x,
                    index_y,
                    obj["x"],
                    obj["y"],
                    obj["size"]
                ):

                    dragged_object = obj
                    obj["dragging"] = True

                    break


        # ----------------------------------------------------
        # DRAG OBJECT
        # ----------------------------------------------------

        if dragged_object is not None:

            obj = dragged_object

            # Move object so its center follows finger
            obj["x"] = index_x - obj["size"] // 2
            obj["y"] = index_y - obj["size"] // 2


            # Keep object inside screen
            obj["x"] = max(
                0,
                min(
                    WIDTH - obj["size"],
                    obj["x"]
                )
            )

            obj["y"] = max(
                0,
                min(
                    HEIGHT - obj["size"],
                    obj["y"]
                )
            )


            # ------------------------------------------------
            # RELEASE OBJECT
            # ------------------------------------------------

            if not is_pinching:

                obj["dragging"] = False

                placed_correctly = False

                for target in targets:

                    if check_placement(
                        obj,
                        target
                    ):

                        # Snap to target
                        obj["x"] = (
                            target["x"]
                            + target["size"] // 2
                            - obj["size"] // 2
                        )

                        obj["y"] = (
                            target["y"]
                            + target["size"] // 2
                            - obj["size"] // 2
                        )

                        obj["placed"] = True

                        score += 1

                        placed_correctly = True

                        break


                # Wrong location
                if not placed_correctly:

                    # Return object to original area
                    obj["x"] = random.randint(
                        100,
                        WIDTH - 200
                    )

                    obj["y"] = random.randint(
                        100,
                        280
                    )


                dragged_object = None


    # ========================================================
    # Draw targets
    # ========================================================

    for target in targets:

        x = target["x"]
        y = target["y"]
        size = target["size"]
        color = target["color"]


        # Target background
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x, y),
            (x + size, y + size),
            color,
            -1
        )

        # Transparency
        cv2.addWeighted(
            overlay,
            0.25,
            frame,
            0.75,
            0,
            frame
        )


        # Target border
        cv2.rectangle(
            frame,
            (x, y),
            (x + size, y + size),
            color,
            4
        )


        # Target text
        cv2.putText(
            frame,
            target["name"],
            (x - 5, y + size + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # ========================================================
    # Draw draggable objects
    # ========================================================

    for obj in objects:

        if obj["placed"]:
            continue


        x = obj["x"]
        y = obj["y"]
        size = obj["size"]
        color = obj["color"]


        # Object
        cv2.rectangle(
            frame,
            (x, y),
            (x + size, y + size),
            color,
            -1
        )


        # Border
        cv2.rectangle(
            frame,
            (x, y),
            (x + size, y + size),
            (255, 255, 255),
            3
        )


        # Object name
        text_size = cv2.getTextSize(
            obj["name"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2
        )[0]


        text_x = (
            x +
            (size - text_size[0]) // 2
        )

        text_y = (
            y +
            (size + text_size[1]) // 2
        )


        cv2.putText(
            frame,
            obj["name"],
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )


    # ========================================================
    # Header
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (WIDTH, 70),
        (30, 30, 30),
        -1
    )


    cv2.putText(
        frame,
        "VIRTUAL DRAG & DROP",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"SCORE: {score}/{TOTAL_OBJECTS}",
        (WIDTH - 250, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # Instructions
    # ========================================================

    cv2.putText(
        frame,
        "Pinch thumb + index finger to drag",
        (30, HEIGHT - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # Game completed
    # ========================================================

    if score == TOTAL_OBJECTS:

        game_completed = True

        cv2.rectangle(
            frame,
            (300, 250),
            (980, 450),
            (30, 30, 30),
            -1
        )


        cv2.putText(
            frame,
            "GAME COMPLETED!",
            (430, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 0),
            3
        )


        cv2.putText(
            frame,
            "Press R to restart",
            (470, 390),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


    # ========================================================
    # Display
    # ========================================================

    cv2.imshow(
        "Virtual Drag and Drop Game",
        frame
    )


    # ========================================================
    # Keyboard controls
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # ESC → exit
    if key == 27:
        break


    # R → restart
    if key == ord("r"):

        objects = create_objects()

        score = 0

        dragged_object = None

        game_completed = False


# ============================================================
# Cleanup
# ============================================================

cap.release()

cv2.destroyAllWindows()

hands.close()