from agent.config import get_config
from vision.inference import SurgicalTripletRecognizer


# ============================================================
# Global model and cache
# ============================================================

_recognizer = None
_vision_cache = {}


def _create_recognizer():
    """
    创建视觉模型。

    模型、设备、标签路径和阈值均来自统一配置。
    """

    config = get_config()
    vision = config.section("vision")

    return SurgicalTripletRecognizer(
        checkpoint_path=str(config.path("model_checkpoint")),
        device=vision.get("device", "cpu"),
        instrument_names_path=str(config.path("instrument_labels")),
        verb_names_path=str(config.path("verb_labels")),
        target_names_path=str(config.path("target_labels")),
        triplet_classes_path=str(config.path("triplet_labels")),
        threshold=float(vision.get("component_threshold", 0.5)),
        triplet_threshold=float(vision.get("triplet_threshold", 0.3)),
        top_k_triplets=int(vision.get("top_k_triplets", 5))
    )


def get_recognizer():
    """
    Lazy-load Vision Model。

    模型只在第一次调用时加载一次，
    后续直接复用。
    """

    global _recognizer

    # 已经加载过
    if _recognizer is not None:
        return _recognizer

    print(
        "[Vision] Loading model..."
    )

    # 第一次创建
    _recognizer = _create_recognizer()

    return _recognizer


def _get_cache_key(
    image_path: str
):
    """
    使用：
        绝对路径 + 文件修改时间

    作为图像缓存 key。
    """

    import os

    absolute_path = os.path.abspath(image_path)

    if not os.path.exists(
        absolute_path
    ):
        raise FileNotFoundError(
            f"找不到图像：{absolute_path}"
        )

    modified_time = os.path.getmtime(
        absolute_path
    )

    return (
        absolute_path,
        modified_time
    )


def analyze_surgical_image(
    image_path: str
) -> dict:
    """
    分析手术图像。

    同一张且未修改的图像已经分析过时，
    直接返回缓存结果。
    """

    # ========================================================
    # 1. 获取缓存 key
    # ========================================================

    cache_key = _get_cache_key(
        image_path
    )

    # ========================================================
    # 2. Cache hit
    # ========================================================

    if cache_key in _vision_cache:

        print(
            "[Vision] Using cached result."
        )

        return _vision_cache[
            cache_key
        ]

    # ========================================================
    # 3. Cache miss
    # ========================================================

    print(
        "[Vision] Running inference..."
    )

    recognizer = get_recognizer()

    result = recognizer.predict(
        image_path
    )

    # ========================================================
    # 4. Save cache
    # ========================================================

    _vision_cache[
        cache_key
    ] = result

    return result


def clear_vision_cache():
    """
    清空视觉缓存。
    """

    _vision_cache.clear()

    print(
        "[Vision] Cache cleared."
    )
