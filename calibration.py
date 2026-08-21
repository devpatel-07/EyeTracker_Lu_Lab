import glob
import os

import cv2
import numpy as np


CHECKERBOARD = (9, 6)  # internal corners, not squares
SQUARE_SIZE_MM = 5.0
IMAGE_FOLDER = "calibration_images"
OUTPUT_FILE = "camera_calibration.npz"


def main():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []
    imgpoints = []
    image_size = None

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in {IMAGE_FOLDER!r}")

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"READ FAIL: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD)

        if not found:
            print(f"NOT DETECTED: {os.path.basename(path)}")
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        print(f"DETECTED: {os.path.basename(path)}")

        preview = img.copy()
        cv2.drawChessboardCorners(preview, CHECKERBOARD, corners, found)
        cv2.imshow("Detected corners", preview)
        cv2.waitKey(150)

    cv2.destroyAllWindows()

    if len(objpoints) < 10:
        raise RuntimeError(
            f"Only {len(objpoints)} valid images detected. "
            "Use at least 10, preferably 15-30."
        )

    reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    np.savez(
        OUTPUT_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        reprojection_error=reprojection_error,
        image_size=np.array(image_size),
        checkerboard=np.array(CHECKERBOARD),
        square_size_mm=SQUARE_SIZE_MM,
    )

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print("\nValid calibration images:", len(objpoints))
    print("Reprojection error:", reprojection_error)
    print("\nCamera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)
    print("\nUse these values in your eye tracker:")
    print("FX =", fx)
    print("FY =", fy)
    print("CX =", cx)
    print("CY =", cy)
    print(f"\nSaved calibration to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
