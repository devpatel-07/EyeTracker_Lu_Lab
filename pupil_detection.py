import math
from pathlib import Path
from tkinter import Tk, filedialog

import cv2

from binocular_3d_view import Binocular3DView
from eye_video_pipeline import (
    EyeCameraCalibration,
    EyeVideoPipeline,
    read_frame_pair,
    validate_synchronized_captures,
)
from ml_pupil_inference import MLPupilDetector
from video_frame_transform import FrameTransform, TransformedVideoCapture


EYE_CENTER_SEPARATION_MM = 63.0
EYE_RADIUS_MM = 12.0
ENABLE_3D_VIEWER = False
THREE_D_VIEWER_MAX_FPS = 30.0
PUPIL_GRAPH_HISTORY_SECONDS = 10.0
PUPIL_GRAPH_BASELINE_SECONDS = 5.0
MIN_PYE3D_CONFIDENCE = 0.60

PROJECT_DIR = Path(__file__).resolve().parent
ML_MODEL_PATH = PROJECT_DIR / "models" / "pupil_unet_best.pt"
ML_MASK_THRESHOLD = 0.5

# These can point to independent calibration files after both cameras are
# calibrated separately.
LEFT_CALIBRATION_PATH = PROJECT_DIR / "camera_calibration.npz"
RIGHT_CALIBRATION_PATH = PROJECT_DIR / "camera_calibration.npz"
LEFT_VIDEO_ROTATION = "none"
RIGHT_VIDEO_ROTATION = "none"
LEFT_FRAME_ROTATION = "counterclockwise"
LEFT_FRAME_FLIP_HORIZONTAL = False
LEFT_FRAME_FLIP_VERTICAL = False
RIGHT_FRAME_ROTATION = "clockwise"
RIGHT_FRAME_FLIP_HORIZONTAL = False
RIGHT_FRAME_FLIP_VERTICAL = False
LEFT_CAMERA_YAW_DEGREES = 35.0
RIGHT_CAMERA_YAW_DEGREES = -35.0

# Shift x/y independently when an eye is positioned differently in either
# camera. The complete rectangle must remain inside its video frame.
LEFT_EYE_RECT = {"x": 0, "y": 50, "w": 980, "h": 700}
RIGHT_EYE_RECT = {"x": 100, "y": 50, "w": 980, "h": 700}


def access_file(side=""):
    root = Tk()
    root.withdraw()
    label = f"{side} eye " if side else ""
    print(f"Select {label}video file")

    root.update()
    video_path = filedialog.askopenfilename(
        title=f"Select {label.title()}Video File",
        filetypes=[("Video Files", "*.mp4")],
    )
    root.destroy()
    return video_path


def main():
    left_path = access_file("left")
    if not left_path:
        return
    right_path = access_file("right")
    if not right_path:
        return

    left_source = None
    right_source = None
    left_video = None
    right_video = None
    viewer = None
    try:
        left_source = cv2.VideoCapture(left_path)
        right_source = cv2.VideoCapture(right_path)
        left_transform = FrameTransform(
            rotation=LEFT_FRAME_ROTATION,
            flip_horizontal=LEFT_FRAME_FLIP_HORIZONTAL,
            flip_vertical=LEFT_FRAME_FLIP_VERTICAL,
        )
        right_transform = FrameTransform(
            rotation=RIGHT_FRAME_ROTATION,
            flip_horizontal=RIGHT_FRAME_FLIP_HORIZONTAL,
            flip_vertical=RIGHT_FRAME_FLIP_VERTICAL,
        )
        left_video = TransformedVideoCapture(left_source, left_transform)
        right_video = TransformedVideoCapture(right_source, right_transform)
        validate_synchronized_captures(left_video, right_video)

        ml_detector = MLPupilDetector.from_checkpoint(
            ML_MODEL_PATH,
            device="auto",
            threshold=ML_MASK_THRESHOLD,
        )
        print(
            "Using ML pupil detector on:",
            getattr(ml_detector, "device", "configured device"),
        )

        left_calibration = EyeCameraCalibration.load(
            LEFT_CALIBRATION_PATH,
            LEFT_VIDEO_ROTATION,
            left_transform,
        )
        right_calibration = EyeCameraCalibration.load(
            RIGHT_CALIBRATION_PATH,
            RIGHT_VIDEO_ROTATION,
            right_transform,
        )

        left_pipeline = EyeVideoPipeline(
            side="left",
            capture=left_video,
            calibration=left_calibration,
            pupil_detector=ml_detector,
            eye_center_separation_mm=EYE_CENTER_SEPARATION_MM,
            rect=LEFT_EYE_RECT,
            camera_yaw_degrees=LEFT_CAMERA_YAW_DEGREES,
            min_confidence=MIN_PYE3D_CONFIDENCE,
            graph_history_seconds=PUPIL_GRAPH_HISTORY_SECONDS,
            graph_baseline_seconds=PUPIL_GRAPH_BASELINE_SECONDS,
        )
        right_pipeline = EyeVideoPipeline(
            side="right",
            capture=right_video,
            calibration=right_calibration,
            pupil_detector=ml_detector,
            eye_center_separation_mm=EYE_CENTER_SEPARATION_MM,
            rect=RIGHT_EYE_RECT,
            camera_yaw_degrees=RIGHT_CAMERA_YAW_DEGREES,
            min_confidence=MIN_PYE3D_CONFIDENCE,
            graph_history_seconds=PUPIL_GRAPH_HISTORY_SECONDS,
            graph_baseline_seconds=PUPIL_GRAPH_BASELINE_SECONDS,
        )

        if ENABLE_3D_VIEWER:
            viewer = Binocular3DView(
                eye_radius_mm=EYE_RADIUS_MM,
                gaze_ray_length_mm=80.0,
                max_fps=THREE_D_VIEWER_MAX_FPS,
                left_camera_yaw_degrees=LEFT_CAMERA_YAW_DEGREES,
                right_camera_yaw_degrees=RIGHT_CAMERA_YAW_DEGREES,
            )

        print(
            "Left frame size:",
            *left_pipeline.frame_size,
            "calibration:",
            LEFT_CALIBRATION_PATH,
        )
        print(
            "Right frame size:",
            *right_pipeline.frame_size,
            "calibration:",
            RIGHT_CALIBRATION_PATH,
        )
        print("Eye-center separation mm:", EYE_CENTER_SEPARATION_MM)
        print(
            "Camera yaw degrees: left =",
            LEFT_CAMERA_YAW_DEGREES,
            "right =",
            RIGHT_CAMERA_YAW_DEGREES,
        )

        last_model_log_s = {"left": -math.inf, "right": -math.inf}
        while True:
            frame_pair = read_frame_pair(left_video, right_video)
            if frame_pair is None:
                break

            left_output = left_pipeline.process_frame(frame_pair[0])
            right_output = right_pipeline.process_frame(frame_pair[1])

            for side, output in (
                ("left", left_output),
                ("right", right_output),
            ):
                if (
                    output.coordinates is not None
                    and output.timestamp_s - last_model_log_s[side] >= 1.0
                ):
                    coordinates = output.coordinates
                    print(
                        f"{side} local eye center mm:",
                        coordinates.local_eye_center_mm,
                    )
                    print(
                        f"{side} local pupil center mm:",
                        coordinates.local_pupil_center_mm,
                    )
                    print(
                        f"{side} shared eye center mm:",
                        coordinates.shared_eye_center_mm,
                    )
                    print(
                        f"{side} shared pupil center mm:",
                        coordinates.shared_pupil_center_mm,
                    )
                    print(
                        f"{side} shared camera lens mm:",
                        coordinates.shared_camera_lens_mm,
                    )
                    print(
                        f"{side} shared gaze direction:",
                        coordinates.shared_gaze_direction,
                    )
                    print(
                        f"{side} pupil diameter mm:",
                        output.pupil_diameter_mm,
                    )
                    print(
                        f"{side} model confidence:",
                        output.model_result.confidence,
                    )
                    print(
                        f"{side} pye3d update ms:",
                        output.model_result.update_time_ms,
                    )
                    last_model_log_s[side] = output.timestamp_s

            cv2.imshow("left_eye", left_output.display)
            cv2.imshow("right_eye", right_output.display)
            cv2.imshow(
                "left_pupil_dilation",
                left_pipeline.render_graph(),
            )
            cv2.imshow(
                "right_pupil_dilation",
                right_pipeline.render_graph(),
            )
            if viewer is not None and not viewer.update(
                left_output.coordinates,
                right_output.coordinates,
            ):
                viewer = None

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as error:
        print("Could not run binocular eye tracker:", error)
    finally:
        if viewer is not None:
            viewer.close()
        if left_video is not None:
            left_video.release()
        elif left_source is not None:
            left_source.release()
        if right_video is not None:
            right_video.release()
        elif right_source is not None:
            right_source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
