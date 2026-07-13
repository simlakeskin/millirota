"""
MilliRota - Görev 2: Genişletilmiş Kalman Filtresi (EKF)
===========================================================
Q&A'da netleşen kurallara göre:
  - Pozisyon, ilk kareye göre MUTLAK olarak gönderilmeli (translation_x/y/z).
  - GPS sağlıklıyken (health_status=1) sunucudan gelen translation ölçümüyle
    güncelleme (update) yapılır.
  - GPS sağlıksızken (health_status=0, oturumun son ~4 dk'sı, kesinti
    aralıklı da olabilir) ölçüm gelmez -> sadece tahmin (predict, sabit hız
    modeliyle) ile pozisyon dead-reckoning şeklinde sürdürülür.

Durum vektörü: [x, y, z, vx, vy, vz]  (sabit hız modeli)
"""

import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, dt: float = 1 / 7.5,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.05):
        self.dt = dt
        self.initialized = False

        self.x = np.zeros(6)               # [x,y,z,vx,vy,vz]
        self.P = np.eye(6) * 1.0           # belirsizlik kovaryansı

        self.Q = np.eye(6) * process_noise         # süreç gürültüsü
        self.R = np.eye(3) * measurement_noise     # ölçüm gürültüsü

        self.H = np.zeros((3, 6))          # ölçüm modeli: sadece pozisyonu gözlemleriz
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

    # ── Durum geçiş modeli (sabit hız) ──────────────────────────────────────

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        return F

    def predict(self, dt: float = None) -> np.ndarray:
        dt = self.dt if dt is None else dt
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.position

    def update(self, measurement: np.ndarray):
        z = np.asarray(measurement, dtype=float).reshape(3)
        y = z - self.H @ self.x                       # innovasyon
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)       # Kalman kazancı
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def step(self, measurement: np.ndarray) -> np.ndarray:
        """GPS sağlıklıyken: predict + update (ölçümle düzeltme)."""
        if not self.initialized:
            self.x[:3] = np.asarray(measurement, dtype=float).reshape(3)
            self.initialized = True
            return self.position
        self.predict()
        self.update(measurement)
        return self.position

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]
