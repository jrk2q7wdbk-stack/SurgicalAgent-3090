import json
import os
from agent.config import get_config
from copy import deepcopy


# ============================================================
# Path
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

KNOWLEDGE_PATH = str(get_config().path("knowledge_base"))


# ============================================================
# Load Knowledge Base
# ============================================================

with open(
    KNOWLEDGE_PATH,
    "r",
    encoding="utf-8"
) as f:
    KNOWLEDGE_BASE = json.load(f)


# ============================================================
# Basic Information
# ============================================================

KNOWLEDGE_SOURCE = "surgical_knowledge.json"

KNOWLEDGE_VERSION = KNOWLEDGE_BASE.get(
    "version",
    "unknown"
)


# ============================================================
# Normalize Name
# ============================================================

def _normalize_name(name):
    """
    统一知识库查询名称。
    """

    if name is None:
        return ""

    return str(name).strip().lower()


# ============================================================
# Instrument Knowledge
# ============================================================

def get_instrument_knowledge(
    name: str
) -> dict:

    normalized_name = _normalize_name(
        name
    )

    data = (
        KNOWLEDGE_BASE
        .get(
            "instruments",
            {}
        )
        .get(
            normalized_name
        )
    )

    if data is None:
        return {
            "found": False,
            "name": normalized_name,
            "knowledge": None
        }

    return {
        "found": True,
        "name": normalized_name,
        "knowledge": deepcopy(data)
    }


# ============================================================
# Verb Knowledge
# ============================================================

def get_verb_knowledge(
    name: str
) -> dict:

    normalized_name = _normalize_name(
        name
    )

    data = (
        KNOWLEDGE_BASE
        .get(
            "verbs",
            {}
        )
        .get(
            normalized_name
        )
    )

    if data is None:
        return {
            "found": False,
            "name": normalized_name,
            "knowledge": None
        }

    return {
        "found": True,
        "name": normalized_name,
        "knowledge": deepcopy(data)
    }


# ============================================================
# Target Knowledge
# ============================================================

def get_target_knowledge(
    name: str
) -> dict:

    normalized_name = _normalize_name(
        name
    )

    data = (
        KNOWLEDGE_BASE
        .get(
            "targets",
            {}
        )
        .get(
            normalized_name
        )
    )

    if data is None:
        return {
            "found": False,
            "name": normalized_name,
            "knowledge": None
        }

    return {
        "found": True,
        "name": normalized_name,
        "knowledge": deepcopy(data)
    }


# ============================================================
# Triplet Knowledge
# ============================================================

def get_triplet_knowledge(
    instrument: str,
    verb: str,
    target: str
) -> dict:
    """
    查询一个 surgical triplet 对应的知识。

    注意：
    当前 surgical_knowledge.json 中的 triplets
    不是完整的 CholecT50 100 类映射。

    因此这里不会使用 triplets 数组判断
    当前预测 triplet 是否“有效”。

    这里只分别查询：
        instrument
        verb
        target

    并返回可溯源知识证据。
    """

    instrument_name = _normalize_name(
        instrument
    )

    verb_name = _normalize_name(
        verb
    )

    target_name = _normalize_name(
        target
    )

    instrument_result = (
        get_instrument_knowledge(
            instrument_name
        )
    )

    verb_result = (
        get_verb_knowledge(
            verb_name
        )
    )

    target_result = (
        get_target_knowledge(
            target_name
        )
    )

    # --------------------------------------------------------
    # 查询是否全部命中
    # --------------------------------------------------------

    all_found = (
        instrument_result["found"]
        and verb_result["found"]
        and target_result["found"]
    )

    # --------------------------------------------------------
    # 缺失项
    # --------------------------------------------------------

    missing_components = []

    if not instrument_result["found"]:
        missing_components.append(
            "instrument"
        )

    if not verb_result["found"]:
        missing_components.append(
            "verb"
        )

    if not target_result["found"]:
        missing_components.append(
            "target"
        )

    # --------------------------------------------------------
    # Structured Result
    # --------------------------------------------------------

    return {

        # ====================================================
        # Evidence Metadata
        # ====================================================

        "evidence_type":
            "surgical_knowledge",

        "source":
            KNOWLEDGE_SOURCE,

        "knowledge_version":
            KNOWLEDGE_VERSION,

        # ====================================================
        # Query
        # ====================================================

        "query": {
            "instrument":
                instrument_name,

            "verb":
                verb_name,

            "target":
                target_name
        },

        # ====================================================
        # Knowledge
        # ====================================================

        "instrument":
            instrument_result,

        "verb":
            verb_result,

        "target":
            target_result,

        # ====================================================
        # Retrieval Status
        # ====================================================

        "all_found":
            all_found,

        "missing_components":
            missing_components,

        # ====================================================
        # Important Constraint
        # ====================================================

        "interpretation_scope": (
            "The returned information contains general "
            "knowledge about the instrument, verb and target. "
            "It does not prove surgical phase, diagnosis, "
            "complication, or surgeon intent in the current video."
        )
    }


# ============================================================
# Knowledge Base Information
# ============================================================

def get_knowledge_base_info() -> dict:
    """
    返回知识库基本信息。
    """

    instruments = (
        KNOWLEDGE_BASE.get(
            "instruments",
            {}
        )
    )

    verbs = (
        KNOWLEDGE_BASE.get(
            "verbs",
            {}
        )
    )

    targets = (
        KNOWLEDGE_BASE.get(
            "targets",
            {}
        )
    )

    triplets = (
        KNOWLEDGE_BASE.get(
            "triplets",
            []
        )
    )

    return {
        "source":
            KNOWLEDGE_SOURCE,

        "version":
            KNOWLEDGE_VERSION,

        "description":
            KNOWLEDGE_BASE.get(
                "description",
                ""
            ),

        "instrument_count":
            len(instruments),

        "verb_count":
            len(verbs),

        "target_count":
            len(targets),

        "listed_triplet_count":
            len(triplets),

        "triplet_list_complete":
            False,

        "note": (
            "The current triplets list is not the complete "
            "CholecT50 100-class triplet mapping."
        )
    }
