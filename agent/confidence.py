"""Shared confidence policy for RDV predictions and language answers."""

from copy import deepcopy


from agent.config import get_config


_CONFIDENCE_CONFIG = get_config().section("confidence")
LOW_CONFIDENCE_THRESHOLD = float(
    _CONFIDENCE_CONFIG.get("low_threshold", 0.50)
)
HIGH_CONFIDENCE_THRESHOLD = float(
    _CONFIDENCE_CONFIG.get("high_threshold", 0.75)
)


def classify_confidence(confidence):
    """Return a stable three-level confidence label."""

    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 0.0

    if value >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if value >= LOW_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def response_policy(level):
    policies = {
        "high": "normal",
        "medium": "conservative",
        "low": "refuse_definite_judgment",
    }
    return policies.get(level, policies["low"])


def enrich_triplet(triplet):
    """Attach confidence metadata without mutating the input mapping."""

    if not triplet:
        return None

    result = deepcopy(triplet)
    level = classify_confidence(result.get("confidence", 0.0))
    result["confidence_level"] = level
    result["status"] = f"{level}_confidence"
    result["answer_policy"] = response_policy(level)
    return result


def _triplet_text(triplet):
    return "{} - {} - {}".format(
        triplet.get("instrument", "unknown"),
        triplet.get("verb", "unknown"),
        triplet.get("target", "unknown"),
    )


def build_grounded_visual_answer(question, triplet):
    """Build a deterministic, evidence-only answer for perception questions.

    Returning the answer from structured RDV fields prevents the language model
    from silently replacing one of the three labels with a plausible label.
    """

    if not triplet:
        return "当前RDV模型没有返回可用的手术三元组，无法作出确定判断。"

    item = enrich_triplet(triplet)
    confidence = float(item.get("confidence", 0.0))
    level = item["confidence_level"]
    full_triplet = _triplet_text(item)
    question_lower = (question or "").lower()

    if "器械" in question or "instrument" in question_lower:
        subject = f"器械 {item.get('instrument', 'unknown')}"
    elif (
        "目标" in question
        or "组织" in question
        or "target" in question_lower
    ):
        subject = f"作用目标 {item.get('target', 'unknown')}"
    elif "动作" in question or "verb" in question_lower:
        subject = f"动作 {item.get('verb', 'unknown')}"
    else:
        subject = f"三元组 {full_triplet}"

    if level == "high":
        return (
            f"RDV模型高置信度识别到{subject}"
            f"（完整三元组：{full_triplet}，置信度 {confidence:.3f}）。"
        )

    if level == "medium":
        return (
            f"RDV模型以中等置信度提示可能为{subject}"
            f"（完整三元组：{full_triplet}，置信度 {confidence:.3f}）。"
            "该结果应结合相邻帧或人工复核，不宜表述为已确定事实。"
        )

    return (
        "当前RDV证据不足，不能作出确定判断。"
        f"最高分候选为 {full_triplet}（置信度 {confidence:.3f}），"
        "该候选仅供复核，不能当作已确认的手术操作。"
    )


def is_perception_only_question(question):
    """Whether a question can be answered only from the current RDV result."""

    text = (question or "").lower()
    knowledge_markers = (
        "什么意思",
        "含义",
        "用途",
        "知识库",
        "知识图谱",
        "为什么",
        "一般",
        "还能",
        "class",
        "cholect",
    )
    temporal_markers = (
        "视频",
        "秒",
        "最近",
        "上一个事件",
        "时间线",
        "阶段",
        "前后",
    )
    perception_markers = (
        "器械",
        "动作",
        "目标",
        "组织",
        "三元组",
        "triplet",
        "instrument",
        "verb",
        "target",
        "做什么",
    )

    if any(marker in text for marker in knowledge_markers + temporal_markers):
        return False
    return any(marker in text for marker in perception_markers)
