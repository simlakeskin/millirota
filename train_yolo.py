"""
MilliRota - YOLOv11 Model Eğitim Scripti
==========================================
TEKNOFEST 2026 - Aşama 4: Model Eğitimi ve Optimizasyonu

Kullanım:
    python train_yolo.py                    # varsayılan
    python train_yolo.py --epochs 200       # daha uzun eğitim
    python train_yolo.py --model yolo11s    # daha küçük model (Raspberry Pi)
    python train_yolo.py --device cuda      # GPU ile eğitim
"""

import argparse
import os
import sys
from pathlib import Path


def train(
    model_size : str  = "yolo11m",     # n/s/m/l/x
    epochs     : int  = 100,
    imgsz      : int  = 640,
    batch      : int  = 16,
    device     : str  = "cpu",
    data_yaml  : str  = "millirota_dataset/millirota.yaml",
    project    : str  = "millirota_runs",
    name       : str  = "millirota_v1",
    resume     : bool = False,
):
    """YOLOv11 modelini eğit."""

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[HATA] ultralytics yüklü değil.")
        print("Çalıştırın: pip install ultralytics")
        sys.exit(1)

    model_name = f"{model_size}.pt"
    print(f"\n{'='*55}")
    print(f"  MilliRota YOLOv11 Eğitimi")
    print(f"  Model   : {model_name}")
    print(f"  Epoch   : {epochs}")
    print(f"  Batch   : {batch}")
    print(f"  ImgSz   : {imgsz}")
    print(f"  Device  : {device}")
    print(f"{'='*55}\n")

    # Modeli yükle (yoksa otomatik indirir)
    model = YOLO(model_name)

    # Eğitim
    results = model.train(
        data       = data_yaml,
        epochs     = epochs,
        imgsz      = imgsz,
        batch      = batch,
        device     = device,
        project    = project,
        name       = name,
        resume     = resume,
        # Optimizasyon
        optimizer  = "AdamW",
        lr0        = 0.001,
        lrf        = 0.01,
        momentum   = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3,
        # Augmentation
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        degrees    = 10.0,
        translate  = 0.1,
        scale      = 0.5,
        flipud     = 0.1,
        fliplr     = 0.5,
        mosaic     = 1.0,
        mixup      = 0.1,
        # Kayıt
        save       = True,
        save_period = 10,
        plots      = True,
        verbose    = True,
        # Erken durdurma
        patience   = 20,
    )

    # En iyi modeli kaydet
    best_model_path = Path(project) / name / "weights" / "best.pt"
    if best_model_path.exists():
        print(f"\n[✓] En iyi model: {best_model_path}")

        # TensorRT export (Jetson için)
        print("\n[Export] TensorRT formatına dönüştürülüyor...")
        try:
            model.export(format="engine", device=0)
            print("[✓] TensorRT export tamamlandı.")
        except Exception as e:
            print(f"[!] TensorRT export atlandı (CPU ortamı): {e}")
            # ONNX olarak kaydet (evrensel)
            try:
                model.export(format="onnx", opset=12)
                print("[✓] ONNX export tamamlandı.")
            except Exception as e2:
                print(f"[!] ONNX export hatası: {e2}")
    else:
        print(f"[!] Model dosyası bulunamadı: {best_model_path}")

    return results


def validate(model_path: str, data_yaml: str = "millirota_dataset/millirota.yaml"):
    """Eğitilmiş modeli doğrulama seti üzerinde değerlendir."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    print(f"\n  mAP@50   : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall   : {metrics.box.mr:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MilliRota YOLOv11 Eğitimi")
    parser.add_argument("--model",   default="yolo11m",
                        choices=["yolo11n","yolo11s","yolo11m","yolo11l","yolo11x"])
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--data",    default="millirota_dataset/millirota.yaml")
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--val",     default=None, help="Sadece doğrulama: model .pt yolu")
    args = parser.parse_args()

    if args.val:
        validate(args.val, args.data)
    else:
        train(
            model_size = args.model,
            epochs     = args.epochs,
            batch      = args.batch,
            imgsz      = args.imgsz,
            device     = args.device,
            data_yaml  = args.data,
            resume     = args.resume,
        )
