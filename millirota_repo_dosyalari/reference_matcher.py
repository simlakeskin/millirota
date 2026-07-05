"""
MilliRota - Görev 3: Referans Nesne Tespiti Modülü
=====================================================
Soru-cevap oturumunda netleşen kurallara göre tasarlanmıştır:

  - Tek bir referans görseli verilir (nesne bazlı, kırpılmış, doğal arka planlı).
  - Aranan nesne referansla BİREBİR aynı olmalı (benzer ama farklı bir nesne kabul edilmez).
  - Belirli bir [start_frame, end_frame] aralığında aranır; aralık dışına bakılmaz,
    aralıklar çakışmaz (aynı anda sadece 1 referans aktif).
  - Referans/sahne modaliteleri karışık olabilir: RGB referans -> termal sahne,
    uydu/yer görüntüsü referans -> drone sahnesi gibi çapraz-modalite eşleşmeler olabilir.
  - Bu yüzden klasik "eğitilmiş sınıf" tespiti değil, ONE-SHOT NESNE EŞLEŞTİRME
    (template/feature retrieval) gerekiyor: model nesneyi daha önce hiç görmemiş olacak.

Yaklaşım (iki katmanlı, otomatik geçişli):
  1) Özellik tabanlı eşleştirme (SIFT/ORB + FLANN + RANSAC homografi)
     -> Aynı modalite (RGB-RGB) ve dokulu nesnelerde güçlü, ölçek/rotasyona dayanıklı.
  2) Kenar tabanlı çok ölçekli şablon eşleştirme (Canny + normalize edilmiş
     çapraz korelasyon, çoklu ölçek/rotasyon taraması)
     -> Çapraz modalite (RGB referans / termal sahne) durumunda yedek katman;
        özellik eşleştirme yetersiz kaldığında otomatik devreye girer.

Sunucu/şartname entegrasyonu:
  detected_undefined_objects: [{object_id, top_left_x, top_left_y,
                                 bottom_right_x, bottom_right_y}]
  Bu modül bu formata uygun sonuç üretir; object_id, aktif referansın
  kimliğidir (server'dan/takım arayüzünden gelen referans id'si).

NOT: Sunucunun "şu an hangi referans aktif, hangi frame aralığında" bilgisini
hangi alan/endpoint ile verdiği bu oturumda netleşmedi (takım arayüzü Cuma
paylaşılacaktı). Bu modül bunu soyutlayan bir `ReferenceTask` nesnesi
üzerinden alır; gerçek şema netleşince sadece FrameInfo/parse kısmı güncellenecek.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


# ─── Görev tanımı (sunucudan/arayüzden gelecek bilgiyi soyutlar) ──────────────

@dataclass
class ReferenceTask:
    """Tek bir referans nesnenin aktif olduğu frame aralığı."""
    object_id:    int          # sunucudaki referans id'si (object_id olarak gönderilecek)
    reference_path: str        # referans görselinin yerel yolu / indirilen dosya
    start_frame:  int          # aralığın başı (dahil)
    end_frame:    int          # aralığın sonu (dahil)

    def is_active(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame


class ReferenceSessionManager:
    """
    Bir oturumdaki tüm ReferenceTask'leri tutar; verilen frame_index için
    hangi referansın aktif olduğunu söyler. Aralıklar çakışmaz (Q&A'da teyit edildi),
    bu yüzden basit lineer tarama yeterli.
    """

    def __init__(self, tasks: Optional[List[ReferenceTask]] = None):
        self.tasks: List[ReferenceTask] = tasks or []
        self._matchers: dict[int, "ReferenceObjectMatcher"] = {}

    def add_task(self, task: ReferenceTask):
        self.tasks.append(task)

    def active_task_for(self, frame_index: int) -> Optional[ReferenceTask]:
        for t in self.tasks:
            if t.is_active(frame_index):
                return t
        return None

    def get_matcher(self, task: ReferenceTask) -> "ReferenceObjectMatcher":
        """Aynı referans için matcher'ı (özellik çıkarımı dahil) cache'ler."""
        if task.object_id not in self._matchers:
            m = ReferenceObjectMatcher()
            m.load_reference(task.reference_path)
            self._matchers[task.object_id] = m
        return self._matchers[task.object_id]


# ─── Eşleştirme sonucu ─────────────────────────────────────────────────────

@dataclass
class MatchResult:
    found:      bool
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    confidence: float = 0.0
    method:     str = "none"   # "feature" | "template" | "none"


# ─── Ana eşleştirici ────────────────────────────────────────────────────────

class ReferenceObjectMatcher:
    """
    Tek bir referans görseli için one-shot nesne arama.
    Önce özellik tabanlı (SIFT->ORB fallback), yetersizse kenar tabanlı
    çok ölçekli şablon eşleştirmeye düşer.
    """

    def __init__(self,
                 min_feature_matches: int = 10,
                 lowe_ratio: float = 0.75,
                 template_scales: Tuple[float, ...] = (0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0),
                 template_angles: Tuple[float, ...] = (0, 90, 180, 270),
                 template_min_score: float = 0.35):
        self.min_feature_matches = min_feature_matches
        self.lowe_ratio = lowe_ratio
        self.template_scales = template_scales
        self.template_angles = template_angles
        self.template_min_score = template_min_score

        # SIFT varsa onu kullan (daha sağlam), yoksa ORB'a düş
        try:
            self._feat = cv2.SIFT_create()
            self._is_sift = True
        except Exception:
            self._feat = cv2.ORB_create(nfeatures=2000)
            self._is_sift = False

        self.ref_img: Optional[np.ndarray] = None
        self.ref_gray: Optional[np.ndarray] = None
        self.ref_edges: Optional[np.ndarray] = None
        self.ref_kp = None
        self.ref_desc = None
        self.ref_w = 0
        self.ref_h = 0

    # ── Referans yükleme ────────────────────────────────────────────────────

    def load_reference(self, path_or_array):
        img = cv2.imread(path_or_array) if isinstance(path_or_array, str) else path_or_array
        if img is None:
            raise ValueError(f"Referans görseli okunamadı: {path_or_array}")
        self.ref_img = img
        self.ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.ref_h, self.ref_w = self.ref_gray.shape[:2]
        self.ref_edges = self._edge_map(self.ref_gray)
        self.ref_kp, self.ref_desc = self._feat.detectAndCompute(self.ref_gray, None)

    @staticmethod
    def _edge_map(gray: np.ndarray) -> np.ndarray:
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.Canny(g, 50, 150)

    # ── Ana arama fonksiyonu ────────────────────────────────────────────────

    def find(self, scene_img: np.ndarray) -> MatchResult:
        if self.ref_gray is None:
            raise RuntimeError("Önce load_reference() çağrılmalı.")

        result = self._match_features(scene_img)
        if result.found:
            return result

        # Çapraz modalite / dokusuz nesne / yetersiz keypoint -> şablon eşleştirme
        return self._match_template(scene_img)

    # ── Katman 1: Özellik tabanlı (SIFT/ORB + homografi) ────────────────────

    def _match_features(self, scene_img: np.ndarray) -> MatchResult:
        if self.ref_desc is None or len(self.ref_kp) < 4:
            return MatchResult(found=False, method="feature")

        scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
        scene_kp, scene_desc = self._feat.detectAndCompute(scene_gray, None)
        if scene_desc is None or len(scene_kp) < 4:
            return MatchResult(found=False, method="feature")

        if self._is_sift:
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            knn = flann.knnMatch(self.ref_desc, scene_desc, k=2)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            knn = bf.knnMatch(self.ref_desc, scene_desc, k=2)

        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)

        if len(good) < self.min_feature_matches:
            return MatchResult(found=False, confidence=len(good) / max(self.min_feature_matches, 1),
                                method="feature")

        src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([scene_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return MatchResult(found=False, method="feature")

        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.min_feature_matches:
            return MatchResult(found=False, confidence=inliers / max(self.min_feature_matches, 1),
                                method="feature")

        corners = np.float32([[0, 0], [self.ref_w, 0],
                               [self.ref_w, self.ref_h], [0, self.ref_h]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        x1, y1 = proj[:, 0].min(), proj[:, 1].min()
        x2, y2 = proj[:, 0].max(), proj[:, 1].max()

        h, w = scene_gray.shape[:2]
        x1, x2 = float(np.clip(x1, 0, w)), float(np.clip(x2, 0, w))
        y1, y2 = float(np.clip(y1, 0, h)), float(np.clip(y2, 0, h))
        if x2 <= x1 or y2 <= y1:
            return MatchResult(found=False, method="feature")

        confidence = min(1.0, inliers / max(len(good), 1) * (inliers / 20))
        return MatchResult(found=True, x1=x1, y1=y1, x2=x2, y2=y2,
                            confidence=round(float(confidence), 3), method="feature")

    # ── Katman 2: Kenar tabanlı çok ölçekli şablon eşleştirme ───────────────

    def _match_template(self, scene_img: np.ndarray) -> MatchResult:
        scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
        scene_edges = self._edge_map(scene_gray)
        sh, sw = scene_edges.shape[:2]

        best_score = -1.0
        best_box = None

        for angle in self.template_angles:
            base = self.ref_edges if angle == 0 else self._rotate(self.ref_edges, angle)
            th, tw = base.shape[:2]

            for scale in self.template_scales:
                rw, rh = int(tw * scale), int(th * scale)
                if rw < 8 or rh < 8 or rw >= sw or rh >= sh:
                    continue
                templ = cv2.resize(base, (rw, rh), interpolation=cv2.INTER_AREA)

                res = cv2.matchTemplate(scene_edges, templ, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > best_score:
                    best_score = max_val
                    best_box = (max_loc[0], max_loc[1], max_loc[0] + rw, max_loc[1] + rh)

        if best_box is None or best_score < self.template_min_score:
            return MatchResult(found=False, confidence=max(best_score, 0.0), method="template")

        x1, y1, x2, y2 = best_box
        return MatchResult(found=True, x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                            confidence=round(float(best_score), 3), method="template")

    @staticmethod
    def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img


# ─── main.py entegrasyon yardımcı fonksiyonu ──────────────────────────────

def run_task3(session_mgr: ReferenceSessionManager,
              frame_index: int,
              scene_img: np.ndarray):
    """
    main.py içindeki ana döngüden çağrılacak tek satırlık entegrasyon noktası.
    Aktif bir referans yoksa boş liste döner (aralık dışı -> aranmaz).
    """
    from server_client import DetectedUndefinedObject

    task = session_mgr.active_task_for(frame_index)
    if task is None:
        return []

    matcher = session_mgr.get_matcher(task)
    result = matcher.find(scene_img)
    if not result.found:
        return []

    return [DetectedUndefinedObject(
        object_id=task.object_id,
        top_left_x=round(result.x1, 2),
        top_left_y=round(result.y1, 2),
        bottom_right_x=round(result.x2, 2),
        bottom_right_y=round(result.y2, 2),
    )]
