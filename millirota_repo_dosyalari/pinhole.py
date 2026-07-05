"""
MilliRota - Pinhole Camera Model & Modalite Yönlendirme
==========================================================
İki sorumluluk:
  1) SmartCameraRouter: gelen karenin RGB mi termal mi olduğunu otomatik
     ayırt eder (Q&A: termal görüntüler RGB formatında ama 3 kanal birbirinin
     aynısı olarak veriliyor -> kanallar arası fark sıfıra yakınsa termaldir).
  2) PinholeCameraModel: MilliRota şartnamesinde geçen kalibrasyon
     parametreleriyle piksel<->ışın dönüşümü (geo-lokalizasyon ön hazırlığı).
"""

import numpy as np
import cv2


class SmartCameraRouter:
    """Kareyi RGB / termal olarak sınıflandırır."""

    def __init__(self, channel_diff_threshold: float = 2.0):
        self.channel_diff_threshold = channel_diff_threshold

    def auto_switch(self, img: np.ndarray) -> str:
        if img is None or img.ndim != 3 or img.shape[2] != 3:
            return "rgb"
        b, g, r = cv2.split(img.astype(np.float32))
        diff = (np.mean(np.abs(b - g)) + np.mean(np.abs(g - r)) + np.mean(np.abs(b - r))) / 3.0
        return "thermal" if diff < self.channel_diff_threshold else "rgb"


class PinholeCameraModel:
    """
    Basit pinhole kamera modeli: intrinsics + İHA pozu kullanarak piksel
    koordinatını yer düzlemine (Z=0 varsayımıyla) izdüşürür.
    Kalibrasyon parametreleri TEKNOFEST örnek veri seti klasöründeki
    kamera kalibrasyon dosyalarından (RGB 1080p / RGB 4K / termal) okunmalı.
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 dist_coeffs=None):
        self.K = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]], dtype=np.float64)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64) if dist_coeffs is not None else np.zeros(5)

    def undistort_point(self, px: float, py: float) -> np.ndarray:
        pts = np.array([[[px, py]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, self.K, self.dist_coeffs, P=self.K)
        return und.reshape(2)

    def pixel_to_ray(self, px: float, py: float) -> np.ndarray:
        """Piksel -> kamera çerçevesinde normalize yön vektörü."""
        x, y = self.undistort_point(px, py)
        ray = np.array([
            (x - self.K[0, 2]) / self.K[0, 0],
            (y - self.K[1, 2]) / self.K[1, 1],
            1.0,
        ])
        return ray / np.linalg.norm(ray)

    def project_to_ground(self, px: float, py: float,
                           cam_position: np.ndarray,
                           cam_rotation: np.ndarray,
                           ground_z: float = 0.0) -> np.ndarray:
        """
        Piksel + İHA pozu (dünya çerçevesinde konum & rotasyon matrisi) ->
        yer düzlemiyle kesişim noktası (dünya koordinatı).
        """
        ray_cam = self.pixel_to_ray(px, py)
        ray_world = cam_rotation @ ray_cam
        if abs(ray_world[2]) < 1e-9:
            return np.array([np.nan, np.nan, ground_z])
        t = (ground_z - cam_position[2]) / ray_world[2]
        return cam_position + t * ray_world
