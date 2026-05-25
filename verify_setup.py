"""Verify Lock-In setup and dependencies."""

import sys
from pathlib import Path


def check_python_version() -> bool:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    mark = "✓" if ok else "✗"
    print(f"{mark} Python {v.major}.{v.minor}.{v.micro} (requires 3.10+)")
    return ok


def check_import(module_name: str, display_name: str = None) -> bool:
    display_name = display_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {display_name}")
        return True
    except ImportError:
        print(f"✗ {display_name} (not installed)")
        return False


def check_camera() -> bool:
    try:
        import cv2
    except ImportError:
        print("✗ Webcam check skipped (cv2 not installed)")
        return False
    cam = cv2.VideoCapture(0)
    ok = cam.isOpened()
    cam.release()
    print(f"{'✓' if ok else '✗'} Webcam accessible")
    return ok


def check_device() -> None:
    try:
        import torch
    except ImportError:
        print("⚠ Skipping device check (torch not installed)")
        return
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("✓ Apple Silicon MPS available")
    else:
        print("⚠ No GPU detected — inference and training will run on CPU")


def check_model() -> bool:
    p = Path("models/distraction_classifier.pt")
    if p.exists():
        size_mb = p.stat().st_size / (1 << 20)
        print(f"✓ Model found: {p} ({size_mb:.1f} MB)")
        return True
    print(f"⚠ Model not found at {p}")
    print("  → python scripts/download_model.py   (fetch v1.0.0 release asset)")
    print("  → or train one (see QUICKSTART.md)")
    return False


def check_config_alignment() -> bool:
    """Check that config.yaml's class order matches checkpoints/metrics.json (if present)."""
    try:
        from src.config import load_config
    except ImportError:
        print("✗ Cannot import src.config — repo isn't pip-installed yet (`pip install -e .`)")
        return False

    try:
        config = load_config("config.yaml")
        print(f"✓ Config classes: {config.classes}")
    except FileNotFoundError:
        print("✗ config.yaml not found")
        return False

    metrics_path = Path("checkpoints/metrics.json")
    if not metrics_path.exists():
        print("⚠ No checkpoints/metrics.json — train a model to enable class-order check")
        return True

    import json
    metrics = json.loads(metrics_path.read_text())
    trained = metrics.get("classes", [])
    if trained and trained != config.classes:
        print(f"✗ Class order mismatch: config={config.classes} vs trained={trained}")
        print("  Predictions will be inverted at inference. Fix config.yaml.")
        return False
    print(f"✓ Config class order matches trained model ({trained})")
    return True


def main() -> int:
    print("=" * 60)
    print("Lock-In — Setup Verification")
    print("=" * 60)
    results = []

    print("\n[Python]")
    results.append(check_python_version())

    print("\n[Required dependencies]")
    for mod, name in [
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("sklearn", "scikit-learn"),
        ("streamlit", "Streamlit"),
    ]:
        results.append(check_import(mod, name))

    print("\n[Optional dependencies]")
    check_import("plyer", "plyer (cross-platform notifications)")
    check_import("win10toast", "win10toast (Windows notifications)")
    check_import("pandas", "pandas")
    check_import("kagglehub", "kagglehub (dataset fetching)")

    print("\n[Hardware]")
    results.append(check_camera())
    check_device()

    print("\n[Project state]")
    check_model()
    results.append(check_config_alignment())

    print("\n" + "=" * 60)
    if all(results):
        print("✓ All critical checks passed.")
        print("\nNext: streamlit run src/dashboard.py   (or)   python -m src.app")
        return 0
    print("✗ Some checks failed. Fix the issues above.")
    print("Hint: pip install -e .")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
