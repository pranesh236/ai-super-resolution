from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt


# ----------------------------- Configuration ----------------------------- #
MODEL_CHOICE = "fsrcnn"  # Supported: "fsrcnn", "edsr", "espcn", "lapsrn"
SCALE_FACTOR = 4
MODEL_PATHS = {
    "edsr": f"EDSR_x{SCALE_FACTOR}.pb",
    "fsrcnn": f"FSRCNN_x{SCALE_FACTOR}.pb",
    "espcn": f"ESPCN_x{SCALE_FACTOR}.pb",
    "lapsrn": f"LapSRN_x{SCALE_FACTOR}.pb",
}
INPUT_IMAGE_PATH = Path("input.jpg")
OUTPUT_AI_PATH = Path("output_high_res.png")
OUTPUT_COMPARISON_PATH = Path("comparison.png")
# ------------------------------------------------------------------------ #


def validate_files(model_path: Path, input_path: Path) -> None:
    """Raise clear errors for missing required files."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'. "
            "Download the correct .pb file and place it next to this script."
        )
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image not found: '{input_path}'. "
            "Add an image named input.jpg (or update INPUT_IMAGE_PATH)."
        )


def run_super_resolution() -> None:
    model_name = MODEL_CHOICE.lower()
    model_path = Path(MODEL_PATHS[model_name])
    validate_files(model_path, INPUT_IMAGE_PATH)

    image = cv2.imread(str(INPUT_IMAGE_PATH))
    if image is None:
        raise ValueError(
            f"Unable to read input image: '{INPUT_IMAGE_PATH}'. "
            "Check that the file is a valid image."
        )

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(model_name, SCALE_FACTOR)

    height, width = image.shape[:2]
    bicubic = cv2.resize(
        image,
        (width * SCALE_FACTOR, height * SCALE_FACTOR),
        interpolation=cv2.INTER_CUBIC,
    )
    ai_result = sr.upsample(image)

    cv2.imwrite(str(OUTPUT_AI_PATH), ai_result)

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bicubic_rgb = cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB)
    ai_rgb = cv2.cvtColor(ai_result, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        ("Original", original_rgb),
        (f"Bicubic Upscale ({SCALE_FACTOR}x)", bicubic_rgb),
        (f"AI Super Resolution ({model_name.upper()} {SCALE_FACTOR}x)", ai_rgb),
    ]

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(str(OUTPUT_COMPARISON_PATH), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"AI-enhanced image saved: {OUTPUT_AI_PATH.resolve()}")
    print(f"Comparison image saved: {OUTPUT_COMPARISON_PATH.resolve()}")


if __name__ == "__main__":
    try:
        run_super_resolution()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
