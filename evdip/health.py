"""Camera health + self-recalibration (ORB feature matching to the calibration frame)."""
import cv2
import numpy as np


class CameraHealth:
    def __init__(self, ref_frame, corners, tol_px=2.5):
        self.orb = cv2.ORB_create(1200)
        g = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        self.kp, self.des = self.orb.detectAndCompute(g, None)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.corners = np.array(corners, np.float32)
        self.lap_ref = cv2.Laplacian(g, cv2.CV_32F).var()
        self.mean_ref = float(g.mean())
        self.tol = tol_px

    def check(self, frame):
        """Returns dict(ok, T, shift_px, inliers, warnings). T maps current -> reference pixels."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        warnings = []
        lap = cv2.Laplacian(g, cv2.CV_32F).var()
        if lap < 0.35 * self.lap_ref:
            warnings.append("image blur / lens fouling suspected")
        if g.mean() < 25:
            warnings.append("lighting failure (image very dark)")
        out = {"ok": True, "T": None, "shift_px": 0.0, "inliers": 0, "warnings": warnings, "calibration_lost": False}
        kp, des = self.orb.detectAndCompute(g, None)
        if des is None or self.des is None or len(kp) < 20:
            warnings.append("not enough image features to verify calibration")
            return out
        good = []
        for pair in self.bf.knnMatch(des, self.des, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        if len(good) < 20:
            warnings.append("calibration could not be verified (few matches)")
            out["calibration_lost"] = len(good) < 8
            return out
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        T, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        if T is None or inl.sum() < 15:
            out["calibration_lost"] = True
            warnings.append("calibration lost: camera view changed too much - recalibrate")
            return out
        # how far did the wall corners move?
        moved = cv2.perspectiveTransform(self.corners.reshape(-1, 1, 2), np.linalg.inv(T)).reshape(-1, 2)
        shift = float(np.mean(np.linalg.norm(moved - self.corners, axis=1)))
        out.update(T=T, shift_px=shift, inliers=int(inl.sum()))
        if shift > self.tol:
            warnings.append(f"camera moved ~{shift:.1f}px: homography re-estimated automatically")
        if shift > 60:
            out["calibration_lost"] = True
            warnings.append("camera moved a lot: recalibrate before trusting coordinates")
        return out
