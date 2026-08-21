import cv2
import os
import sys
from tkinter import Tk, filedialog
import numpy as np
import random
import math
import time
from pathlib import Path

from camera_geometry import (
    distort_points,
    rotated_camera_matrix,
    undistort_ellipse,
)
from eye_radius_calibration import EyeRadiusCalibrator
from ml_pupil_inference import MLPupilDetector
from pupil_dilation_graph import PupilDilationGraph

# Global variables
eye_center = None
rays = []
MAX_RAYS = 100
stored_intersections = []
MAX_INTERSECTIONS = 1500
frame_counter = 0
MIN_ANGLE = 3.0
NUMBER_RAYS = 4
EYE_RADIUS_MM = 12
RADIUS_CALIBRATION_SAMPLES = 300
RADIUS_CALIBRATION_PERCENTILE = 95
PUPIL_GRAPH_HISTORY_SECONDS = 10.0
PUPIL_GRAPH_BASELINE_SECONDS = 5.0
PUPIL_GRAPH_WINDOW_NAME = "pupil_dilation"

# Pupil detector configuration. Set this to False to use the original
# threshold/contour detector without changing the rest of the tracker.
USE_ML_PUPIL_DETECTOR = True
ML_MODEL_PATH = Path(__file__).resolve().parent / "models" / "pupil_unet_best.pt"
ML_MASK_THRESHOLD = 0.5

# Calibration was computed in the camera's raw 1920x1080 landscape orientation.
CALIBRATION_PATH = Path(__file__).resolve().parent / "camera_calibration.npz"
VIDEO_ROTATION = "clockwise"

with np.load(CALIBRATION_PATH) as calibration:
    RAW_CAMERA_MATRIX = calibration["camera_matrix"].astype(np.float64)
    DISTORTION_COEFFICIENTS = calibration["dist_coeffs"].astype(np.float64)
    RAW_IMAGE_SIZE = tuple(int(value) for value in calibration["image_size"])

CAMERA_MATRIX = rotated_camera_matrix(
    RAW_CAMERA_MATRIX,
    RAW_IMAGE_SIZE,
    VIDEO_ROTATION,
)
FX = CAMERA_MATRIX[0, 0]
FY = CAMERA_MATRIX[1, 1]
CX = CAMERA_MATRIX[0, 2]
CY = CAMERA_MATRIX[1, 2]


def create_radius_calibrator(rect):
    return EyeRadiusCalibrator(
        target_samples=RADIUS_CALIBRATION_SAMPLES,
        percentile=RADIUS_CALIBRATION_PERCENTILE,
        min_radius_px=20,
        max_radius_px=min(rect["w"], rect["h"]),
    )


def resolve_graph_timestamp(video, fallback_start_s, previous_timestamp_s):
    capture_timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
    if math.isfinite(capture_timestamp_ms) and capture_timestamp_ms >= 0:
        capture_timestamp_s = capture_timestamp_ms / 1000.0
        if (
            previous_timestamp_s is None
            or capture_timestamp_s > previous_timestamp_s
        ):
            return capture_timestamp_s

    fallback_timestamp_s = time.monotonic() - fallback_start_s
    if (
        previous_timestamp_s is not None
        and fallback_timestamp_s <= previous_timestamp_s
    ):
        return math.nextafter(previous_timestamp_s, math.inf)
    return fallback_timestamp_s


# Method to detect pupil center and size
def detect_pupil(frame, rect, debug=True):
    x = max(0, rect["x"])
    y = max(0, rect["y"])
    w = min(rect["w"], frame.shape[1] - x)
    h = min(rect["h"], frame.shape[0] - y)

    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Fast smoothing. Median helps with small LED/specular dots.
    gray = cv2.medianBlur(gray, 5)

    # Adaptive-ish dark threshold based on ROI brightness.
    # Pupil should be among the darkest pixels.
    dark_level = np.percentile(gray, 5)
    threshold_value = int(np.clip(dark_level + 0, 0, dark_level+5))

    _, mask = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Fill LED reflection holes and reconnect split pupil pieces.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Remove tiny dark noise from eyelashes/reflections.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_score = -1

    for cnt in contours:

        if not contour_fully_inside_rect(cnt, rect, margin=5):
            continue

        area = cv2.contourArea(cnt)

        if area < 150:
            continue

        ellipse = cv2.fitEllipse(cnt)
        contrast = ellipse_boundary_contrast(gray, ellipse)

        bx, by, bw, bh = cv2.boundingRect(cnt)

        # Reject huge regions and skinny eyelash-like regions.
        if bw < 8 or bh < 8:
            continue
        if bw > w * 0.6 or bh > h * 0.6:
            continue

        aspect = bw / float(bh)
        if aspect < 0.35 or aspect > 2.8:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)

        # Mean darkness inside contour. Real pupil should be dark.
        cnt_mask = np.zeros_like(gray)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(gray, mask=cnt_mask)[0]

        # Higher score is better.
        # Favor large, round-ish, dark blobs.
        score = area * (0.5 + circularity) + contrast * 500 - mean_intensity * 8

        if score > best_score:
            best_score = score
            best_cnt = cnt

    display = frame.copy()
    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if best_cnt is None or len(best_cnt) < 5:
        print("Pupil not detected")
        if debug:
            return mask, 0, True
        return display, 0, True

    try:
        ellipse = cv2.fitEllipse(best_cnt)

        # Shift ellipse coordinates from ROI coordinates back to full-frame coordinates.
        (cx, cy), axes, angle = ellipse
        ellipse = ((cx + x, cy + y), axes, angle)

        cv2.ellipse(display, ellipse, (0, 255, 0), 2)

        if debug:
            debug_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.ellipse(debug_bgr, ((cx, cy), axes, angle), (0, 255, 0), 2)
            return debug_bgr, ellipse, False

        return display, ellipse, False

    except cv2.error:
        print("Contour shape distortion")
        if debug:
            return mask, 0, True
        return display, 0, True
    
def contour_fully_inside_rect(cnt, rect, margin=0):
    x, y, w, h = cv2.boundingRect(cnt)

    left = rect["x"] + margin
    top = rect["y"] + margin
    right = rect["x"] + rect["w"] - margin
    bottom = rect["y"] + rect["h"] - margin

    return (
        x >= left and
        y >= top and
        x + w <= right and
        y + h <= bottom
    )

def ellipse_boundary_contrast(gray, ellipse, samples=48, offset=4):
    (cx, cy), (w, h), angle_deg = ellipse

    if w <= 0 or h <= 0:
        return -999

    a = w / 2
    b = h / 2
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    inside_vals = []
    outside_vals = []

    height, width = gray.shape[:2]

    for i in range(samples):
        t = 2 * math.pi * i / samples

        # Point on ellipse in local coordinates.
        lx = a * math.cos(t)
        ly = b * math.sin(t)

        # Rotate into image coordinates.
        x = cx + lx * cos_t - ly * sin_t
        y = cy + lx * sin_t + ly * cos_t

        # Approximate outward direction from center.
        dx = x - cx
        dy = y - cy
        norm = math.hypot(dx, dy)

        if norm == 0:
            continue

        ux = dx / norm
        uy = dy / norm

        xi = int(round(x - offset * ux))
        yi = int(round(y - offset * uy))
        xo = int(round(x + offset * ux))
        yo = int(round(y + offset * uy))

        if (
            0 <= xi < width and 0 <= yi < height and
            0 <= xo < width and 0 <= yo < height
        ):
            inside_vals.append(gray[yi, xi])
            outside_vals.append(gray[yo, xo])

    if len(inside_vals) < samples * 0.5:
        return -999

    inside_mean = float(np.mean(inside_vals))
    outside_mean = float(np.mean(outside_vals))

    # Pupil should be darker inside, brighter outside.
    return outside_mean - inside_mean

# Determine pupil diameter
def calculate_pupil_diameter(major_axis_px, pupil_depth):

    effective_focal_length = (FX + FY) / 2
    pupil_diameter_mm = major_axis_px * pupil_depth / effective_focal_length

    return pupil_diameter_mm

# find the intersection of 2 lines
def find_line_intersection(ellipse1, ellipse2):
    
    # Unpack the ellipses
    (cx1, cy1), (_, major_axis1), angle1 = ellipse1
    (cx2, cy2), (_, major_axis2), angle2 = ellipse2

    # Convert orientation angles from degrees to radians for trigonometric functions
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)

    # Compute 2D direction vectors (dx, dy) for both projection lines using trig.
    # The length of the vector is scaled by half the major axis length (the radius).
    dx1, dy1 = (major_axis1 / 2) * np.cos(angle1_rad), (
        major_axis1 / 2
    ) * np.sin(angle1_rad)
    dx2, dy2 = (major_axis2 / 2) * np.cos(angle2_rad), (
        major_axis2 / 2
    ) * np.sin(angle2_rad)

    #
    # Matrix A contains the direction vectors:
    # [ dx1   -dx2 ]
    # [ dy1   -dy2 ]
    A = np.array([[dx1, -dx2], [dy1, -dy2]])

    # Vector B represents the distance offset between the two ellipse centers:
    B = np.array([cx2 - cx1, cy2 - cy1])

    # Check the determinant of Matrix A.
    # If the determinant is 0, the lines are parallel and will never cross.
    if np.linalg.det(A) == 0:
        return None

    # Solve the system of linear equations A * T = B
    t1, t2 = np.linalg.solve(A, B)

    # Use the parameter t1 to find the exact intersection point along Line 1
    intersection_x = cx1 + t1 * dx1
    intersection_y = cy1 + t1 * dy1

    return (int(intersection_x), int(intersection_y))


def prune_intersections(intersections, maximum_intersections):
    """Removes the oldest intersections to ensure only the last M intersections remain."""
    if len(intersections) <= maximum_intersections:
        return intersections  # No need to prune if within the limit

    # Keep only the last M (most recent) intersections
    return intersections[-maximum_intersections:]


def compute_average_intersection(
    frame, ray_lines, number_lines, total_lines, minimum_angle_degrees
):
    """Selects `number_lines` random lines from the history, computes their intersections,

    verifies that they agree within a pixel limit, stores them, and prunes older data.

    Parameters:
    -----------
    frame : np.ndarray
        The current camera image frame (used to verify bounds).
    ray_lines : list
        The list of all pupil ellipse rays accumulated over the last frames.
    number_lines : int
        How many rays to randomly sample and cross this frame (e.g., 4).
    total_lines : int
        The maximum limit for the history buffer size (e.g., 1500).
    minimum_angle_degrees : float
        Minimum angular difference between rays required to compute a valid cross.

    Returns:
    --------
    (avg_x, avg_y) : tuple of ints
        The newly calculated running average of the 3D eye center projection,
        or (0, 0) if calculation was not possible.
    """
    pixel_limit = 30  # Max pixel distance allowed between intersections to accept them
    angle_threshold = 5  # degrees (to ensure rays aren't practically parallel)

    global stored_intersections

    # Ensure we have enough historical rays to calculate intersections
    if len(ray_lines) < 2 or number_lines < 2:
        return (0, 0)

    height, width = frame.shape[:2]

    # Randomly sample a subset of rays from history to test
    selected_lines = random.sample(ray_lines, min(number_lines, len(ray_lines)))

    intersections = []

    # Loop through and intersect adjacent pairs in our random selection
    for i in range(len(selected_lines) - 1):
        line1 = selected_lines[i]
        line2 = selected_lines[i + 1]

        angle1 = line1[2]
        angle2 = line2[2]

        # Ensure the two rays have a high enough angle separation to intersect cleanly
        if abs(angle1 - angle2) >= minimum_angle_degrees:
            # (Note: find_line_intersection is called here)
            intersection = find_line_intersection(line1, line2)

            # Ensure the calculated intersection point falls within actual image boundaries
            if (
                intersection
                and (0 <= intersection[0] < width)
                and (0 <= intersection[1] < height)
            ):
                intersections.append(intersection)

    # If no valid intersections were found in this frame's sample, return previous eye center
    if not intersections:
        if len(stored_intersections):
            avg_x = np.mean([pt[0] for pt in stored_intersections])
            avg_y = np.mean([pt[1] for pt in stored_intersections])
            return (int(avg_x), int(avg_y))
        else:
            return (0, 0)

    #  Verification - Ensure the newly found intersection points agree with each other
    accept = True
    if len(intersections) >= 2:
        for i in range(len(intersections)):
            for j in range(i + 1, len(intersections)):
                # Calculate Euclidean distance between calculated intersection points
                dx = intersections[i][0] - intersections[j][0]
                dy = intersections[i][1] - intersections[j][1]
                distance = (dx * dx + dy * dy) ** 0.5

                # Guard 1: Reject if the intersections are scattered too far apart
                if distance > pixel_limit:
                    accept = False
                    break

                # Guard 2: Reject if the angles of the lines are too similar
                angle_i = selected_lines[i][2]
                angle_j = selected_lines[j][2]
                if abs(angle_i - angle_j) < angle_threshold:
                    accept = False
                    break
            if not accept:
                break

    # Append valid intersections to the persistent rolling buffer
    if accept:
        stored_intersections.extend(intersections)

    #  Prune our global history to prevent memory leaking beyond `total_lines`
    if len(stored_intersections) > total_lines:
        stored_intersections = prune_intersections(
            stored_intersections, total_lines
        )

    # Calculate the final coordinate of the eye center
    if not stored_intersections:
        if len(stored_intersections):
            avg_x = np.mean([pt[0] for pt in stored_intersections])
            avg_y = np.mean([pt[1] for pt in stored_intersections])
            return (int(avg_x), int(avg_y))
        else:
            return (0, 0)

    # Take the mean of all stored coordinates to yield the final, stable 2D coordinate
    avg_x = np.mean([pt[0] for pt in stored_intersections])
    avg_y = np.mean([pt[1] for pt in stored_intersections])

    if np.isnan(avg_x) or np.isnan(avg_y):
        if len(stored_intersections):
            avg_x = np.mean([pt[0] for pt in stored_intersections])
            avg_y = np.mean([pt[1] for pt in stored_intersections])
            return (int(avg_x), int(avg_y))
        else:
            return (0, 0)

    return (int(avg_x), int(avg_y))

# Calculate distance from center of eye to pupil
def distance_to_pupil_outer_edge(eye_center, pupil_ellipse):
    """Calculates the 2D pixel distance from the estimated eyeball center

    to the furthest outer boundary of the pupil ellipse along the displacement vector.

    Parameters:
    -----------
    eye_center : tuple of (float, float)
        The coordinates of the estimated eyeball center (e.g., model_center_average).
    pupil_ellipse : tuple
        The OpenCV ellipse representation of the pupil:
        ((cx, cy), (width, height), angle_degrees)

    Returns:
    --------
    total_distance : float or None
        The total distance from the eyeball center to the outer edge of the ellipse,
        or None if calculation is mathematically impossible (e.g., zero divisions).
    """
    # Unpack the ellipse parameters
    pupil_center, axes, angle_degrees = pupil_ellipse
    semi_axis_x = axes[0] / 2
    semi_axis_y = axes[1] / 2

    # Step 1: Compute displacement vector from eyeball center to pupil center
    direction_x = pupil_center[0] - eye_center[0]
    direction_y = pupil_center[1] - eye_center[1]

    # Calculate 2D Euclidean distance between the centers
    center_distance = math.hypot(direction_x, direction_y)

    # Safety checks to prevent zero division
    if center_distance == 0 or semi_axis_x <= 0 or semi_axis_y <= 0:
        return None

    # Step 2: Normalize the displacement direction vector
    unit_x = direction_x / center_distance
    unit_y = direction_y / center_distance

    # Convert the ellipse's rotation angle to radians
    angle_radians = math.radians(angle_degrees)
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)

    # Step 3: Rotate the 2D unit vector into the local coordinate space of the ellipse.
    # This aligns our lookup vector with the ellipse's major/minor axes.
    local_x = cosine * unit_x + sine * unit_y
    local_y = -sine * unit_x + cosine * unit_y

    # Step 4: Use the mathematical equation of an ellipse to find the boundary offset
    # Ellipse equation: (x/a)^2 + (y/b)^2 = 1
    # For a unit direction vector, the distance to the edge is: 1 / sqrt((x/a)^2 + (y/b)^2)
    edge_offset = 1 / math.sqrt(
        (local_x / semi_axis_x) ** 2 + (local_y / semi_axis_y) ** 2
    )

    # Step 5: The total distance is the center-to-center distance plus the edge offset
    return center_distance + edge_offset

# convert the eye center into 3D and in millimeters
def compute_3D_eye_center(eye_center, optical_center, calibrated_radius_px):

    effective_focal_length = (FX + FY) / 2
    Zc = EYE_RADIUS_MM * effective_focal_length / calibrated_radius_px
    Xc = (eye_center[0] - optical_center[0]) * Zc / FX
    Yc = (eye_center[1] - optical_center[1]) * Zc / FY

    return (Xc, Yc, Zc)

def compute_3D_pupil_center(pupil_center, eye_center_3D, optical_center):

    (Xc, Yc, Zc) = eye_center_3D

    # Determine slope of ray from camera (0,0,0) to pupil center
    slope_x = (pupil_center[0] - optical_center[0]) / FX
    slope_y = (pupil_center[1] - optical_center[1]) / FY

    # Use quadratic formula to determine depth at which ray intersects eye sphere
    A = slope_x**2 + slope_y**2 + 1
    B = -2 * (slope_x * Xc + slope_y * Yc + Zc)
    C = Xc**2 + Yc**2 + Zc**2 - EYE_RADIUS_MM**2
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return None
    Zp = (-B - math.sqrt(discriminant)) / (2 * A) # Use smaller Zp value to get first intersection of ray and sphere

    # Calculate X and Y coordinates of pupil 
    Xp = slope_x * Zp
    Yp = slope_y * Zp

    return (Xp, Yp, Zp)


# Access .mp4 file
def access_file():

    root = Tk()
    root.withdraw() # Hide main window

    # Allowed file types
    file_types = [("Video Files", "*.mp4")]

    print("Select video file")

    # Open file selector
    root.update()
    video_path = filedialog.askopenfilename(
        title="Select a Video File", filetypes=file_types
    )

    root.destroy()
    return video_path


def main():
    global rays
    global frame_counter
    global eye_center

    path = access_file()
    video = cv2.VideoCapture(path)

    ml_detector = None
    if USE_ML_PUPIL_DETECTOR:
        try:
            ml_detector = MLPupilDetector.from_checkpoint(
                ML_MODEL_PATH,
                device="auto",
                threshold=ML_MASK_THRESHOLD,
            )
            print("Using ML pupil detector on:", ml_detector.device)
        except (FileNotFoundError, RuntimeError, KeyError) as error:
            print("Could not load ML pupil detector:", error)
            print("Falling back to the classical pupil detector")

    # Get video height and width and determine optical center
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_width, raw_height = RAW_IMAGE_SIZE
    if VIDEO_ROTATION == "none":
        expected_size = (raw_width, raw_height)
    else:
        expected_size = (raw_height, raw_width)
    if (width, height) != expected_size:
        raise ValueError(
            f"Video size {(width, height)} does not match calibrated "
            f"{VIDEO_ROTATION} size {expected_size}"
        )

    optical_center = (CX, CY)
    print("Frame size:", width, "x", height)
    print("Using intrinsics: FX =", FX, "FY =", FY, "CX =", CX, "CY =", CY)
    # Set coordinates of eye-enclosing rectangle
    rect = {'x':100, 'y':100, 'w':width-100, 'h':600}
    radius_calibrator = create_radius_calibrator(rect)
    dilation_graph = PupilDilationGraph(
        history_seconds=PUPIL_GRAPH_HISTORY_SECONDS,
        baseline_seconds=PUPIL_GRAPH_BASELINE_SECONDS,
    )
    graph_fallback_start_s = time.monotonic()
    graph_timestamp_s = None
    graph_started = False
    print(
        "Eye-radius calibration: look slowly left, right, up, and down until "
        f"{RADIUS_CALIBRATION_SAMPLES} valid samples are collected."
    )

    while True:
        # read frame
        ongoing, frame = video.read()
        # break if video has ended
        if ongoing == False:
            break

        graph_timestamp_s = resolve_graph_timestamp(
            video,
            graph_fallback_start_s,
            graph_timestamp_s,
        )

        # Detect pupil
        if ml_detector is not None:
            pupil_frame, ellipse, blink = ml_detector.detect(
                frame, rect, debug=False
            )
        else:
            pupil_frame, ellipse, blink = detect_pupil(
                frame, rect, debug=False
            )

        geometry_ellipse = None
        if blink == False:
            geometry_ellipse = undistort_ellipse(
                ellipse,
                RAW_CAMERA_MATRIX,
                DISTORTION_COEFFICIENTS,
                RAW_IMAGE_SIZE,
                VIDEO_ROTATION,
            )
            rays.append(geometry_ellipse)
            if len(rays) > MAX_RAYS:
                rays = rays[-MAX_RAYS:]
        frame_counter += 1

        # Geometry is computed in undistorted rotated-image coordinates.
        eye_center = compute_average_intersection(frame, rays, NUMBER_RAYS, MAX_INTERSECTIONS, MIN_ANGLE)
        if eye_center != (0, 0):
            display_eye_center = distort_points(
                np.array([eye_center], dtype=np.float32),
                RAW_CAMERA_MATRIX,
                DISTORTION_COEFFICIENTS,
                RAW_IMAGE_SIZE,
                VIDEO_ROTATION,
            )[0]
            display_eye_center = tuple(int(round(value)) for value in display_eye_center)
            cv2.circle(pupil_frame, display_eye_center, 4, (255,0,0), -1)

        # Collect a robust eye-radius scale and freeze it after calibration.
        if geometry_ellipse is not None and eye_center != (0, 0) and frame_counter > 30:
            center_to_pupil = distance_to_pupil_outer_edge(
                eye_center, geometry_ellipse
            )
            accepted = radius_calibrator.add(center_to_pupil)
            if accepted and (
                radius_calibrator.sample_count % 30 == 0
                or radius_calibrator.ready
            ):
                print(
                    "Eye-radius calibration:",
                    radius_calibrator.sample_count,
                    "/",
                    RADIUS_CALIBRATION_SAMPLES,
                )
            if accepted and radius_calibrator.ready:
                print(
                    "Eye-radius calibration complete. Radius =",
                    radius_calibrator.radius_px,
                    "pixels",
                )

        pupil_diameter = None
        if radius_calibrator.ready and not graph_started:
            dilation_graph.start(graph_timestamp_s)
            graph_started = True

        if radius_calibrator.ready and geometry_ellipse is not None:
            eye_center_3D = compute_3D_eye_center(
                eye_center,
                optical_center,
                radius_calibrator.radius_px,
            )
            pupil_center_3D = compute_3D_pupil_center(
                geometry_ellipse[0], eye_center_3D, optical_center
            )

            if pupil_center_3D is not None:
                pupil_diameter = calculate_pupil_diameter(
                    max(geometry_ellipse[1]), pupil_center_3D[2]
                )
                print("major_axis_px:", max(geometry_ellipse[1]))
                print("Zp:", pupil_center_3D[2])
                print("diameter_mm:", pupil_diameter)
                print("eye center depth:", eye_center_3D[2])

        if graph_started:
            dilation_graph.add_sample(graph_timestamp_s, pupil_diameter)

        # Show frame
        cv2.imshow('pupil_detection', pupil_frame)
        cv2.imshow(PUPIL_GRAPH_WINDOW_NAME, dilation_graph.render())

        # 30 ms between frames, press Q to exit
        key = cv2.waitKey(30)
        if key == 113:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
