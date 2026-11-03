import json
import os
from copy import deepcopy
from agent.config import get_config


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

GRAPH_PATH = str(get_config().path("knowledge_graph"))


class SurgicalGraphKnowledgeTool:

    def __init__(self, graph_path=GRAPH_PATH):

        with open(
            graph_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.graph = json.load(f)

        self.nodes = self.graph["nodes"]
        self.edges = self.graph["edges"]

        # --------------------------------------------------
        # 建立索引，避免每次查询都扫描整个 JSON
        # --------------------------------------------------

        self.node_by_id = {
            node["id"]: node
            for node in self.nodes
        }

        self.triplet_by_class_id = {}

        self.triplets_by_instrument = {}
        self.triplets_by_verb = {}
        self.triplets_by_target = {}

        self.instrument_by_name = {}
        self.verb_by_name = {}
        self.target_by_name = {}

        for node in self.nodes:

            node_type = node.get("type")

            if node_type == "instrument":

                self.instrument_by_name[
                    node["name"]
                ] = node

            elif node_type == "verb":

                self.verb_by_name[
                    node["name"]
                ] = node

            elif node_type == "target":

                self.target_by_name[
                    node["name"]
                ] = node

            elif node_type == "triplet":

                class_id = node["class_id"]

                self.triplet_by_class_id[
                    class_id
                ] = node

                instrument = node["instrument"]
                verb = node["verb"]
                target = node["target"]

                self.triplets_by_instrument.setdefault(
                    instrument,
                    []
                ).append(node)

                self.triplets_by_verb.setdefault(
                    verb,
                    []
                ).append(node)

                self.triplets_by_target.setdefault(
                    target,
                    []
                ).append(node)

    # ======================================================
    # Helper
    # ======================================================

    @staticmethod
    def _filter_null(triplets, include_null=False):

        if include_null:
            return triplets

        return [
            triplet
            for triplet in triplets
            if not triplet.get("is_null", False)
        ]

    # ======================================================
    # Query 1
    # 根据 IVT class ID 查询
    # ======================================================

    def query_triplet(
        self,
        class_id,
        include_null=True
    ):

        try:
            class_id = int(class_id)
        except (TypeError, ValueError):
            return {
                "found": False,
                "error": "class_id 必须是整数"
            }

        triplet = self.triplet_by_class_id.get(
            class_id
        )

        if triplet is None:

            return {
                "found": False,
                "class_id": class_id,
                "error": "未找到该 IVT class"
            }

        if (
            not include_null
            and triplet.get("is_null", False)
        ):

            return {
                "found": False,
                "class_id": class_id,
                "error": "该 IVT class 为 null triplet"
            }

        return {
            "found": True,
            "triplet": deepcopy(triplet)
        }

    # ======================================================
    # Query 2
    # 查询某个器械的所有有效 Triplet
    # ======================================================

    def query_by_instrument(
        self,
        instrument,
        include_null=False
    ):

        if instrument not in self.instrument_by_name:

            return {
                "found": False,
                "instrument": instrument,
                "error": "未知 instrument",
                "available": sorted(
                    self.instrument_by_name.keys()
                )
            }

        triplets = self.triplets_by_instrument.get(
            instrument,
            []
        )

        triplets = self._filter_null(
            triplets,
            include_null
        )

        return {
            "found": True,
            "instrument": instrument,
            "instrument_info": deepcopy(
                self.instrument_by_name[instrument]
            ),
            "count": len(triplets),
            "triplets": deepcopy(triplets)
        }

    # ======================================================
    # Query 3
    # 查询某个动作的所有有效 Triplet
    # ======================================================

    def query_by_verb(
        self,
        verb,
        include_null=False
    ):

        if verb not in self.verb_by_name:

            return {
                "found": False,
                "verb": verb,
                "error": "未知 verb",
                "available": sorted(
                    self.verb_by_name.keys()
                )
            }

        triplets = self.triplets_by_verb.get(
            verb,
            []
        )

        triplets = self._filter_null(
            triplets,
            include_null
        )

        return {
            "found": True,
            "verb": verb,
            "verb_info": deepcopy(
                self.verb_by_name[verb]
            ),
            "count": len(triplets),
            "triplets": deepcopy(triplets)
        }

    # ======================================================
    # Query 4
    # 查询某个 target 的所有有效 Triplet
    # ======================================================

    def query_by_target(
        self,
        target,
        include_null=False
    ):

        if target not in self.target_by_name:

            return {
                "found": False,
                "target": target,
                "error": "未知 target",
                "available": sorted(
                    self.target_by_name.keys()
                )
            }

        triplets = self.triplets_by_target.get(
            target,
            []
        )

        triplets = self._filter_null(
            triplets,
            include_null
        )

        return {
            "found": True,
            "target": target,
            "target_info": deepcopy(
                self.target_by_name[target]
            ),
            "count": len(triplets),
            "triplets": deepcopy(triplets)
        }

    # ======================================================
    # Query 5
    # 组合条件查询
    #
    # instrument / verb / target 均可选
    # ======================================================

    def query_relation(
        self,
        instrument=None,
        verb=None,
        target=None,
        include_null=False
    ):

        results = []

        for triplet in self.triplet_by_class_id.values():

            if (
                not include_null
                and triplet.get("is_null", False)
            ):
                continue

            if (
                instrument is not None
                and triplet["instrument"] != instrument
            ):
                continue

            if (
                verb is not None
                and triplet["verb"] != verb
            ):
                continue

            if (
                target is not None
                and triplet["target"] != target
            ):
                continue

            results.append(triplet)

        results.sort(
            key=lambda x: x["class_id"]
        )

        return {
            "found": len(results) > 0,
            "query": {
                "instrument": instrument,
                "verb": verb,
                "target": target
            },
            "count": len(results),
            "triplets": deepcopy(results)
        }

    # ======================================================
    # Query 6
    # 查询器械支持哪些动作
    #
    # 这里读取 supports_action 图边，
    # 与 Triplet 数据集共现关系分开。
    # ======================================================

    def query_supported_actions(
        self,
        instrument
    ):

        instrument_node = (
            self.instrument_by_name.get(
                instrument
            )
        )

        if instrument_node is None:

            return {
                "found": False,
                "instrument": instrument,
                "error": "未知 instrument"
            }

        source_id = instrument_node["id"]

        actions = []

        for edge in self.edges:

            if (
                edge["source"] == source_id
                and edge["relation"]
                == "supports_action"
            ):

                verb_node = self.node_by_id.get(
                    edge["target"]
                )

                if verb_node:
                    actions.append(
                        deepcopy(verb_node)
                    )

        return {
            "found": True,
            "instrument": instrument,
            "count": len(actions),
            "actions": actions
        }

    # ======================================================
    # Query 7
    # KG 概况
    # ======================================================

    def get_graph_summary(self):

        return deepcopy(
            self.graph["metadata"]
        )


# ==========================================================
# 全局实例
# ==========================================================

_graph_tool = None


def get_graph_tool():

    global _graph_tool

    if _graph_tool is None:

        print(
            "[KnowledgeGraph] Loading graph..."
        )

        _graph_tool = (
            SurgicalGraphKnowledgeTool()
        )

        print(
            "[KnowledgeGraph] Graph loaded."
        )

    return _graph_tool


# ==========================================================
# Agent-friendly unified interface
# ==========================================================

def query_surgical_graph(
    query_type,
    class_id=None,
    instrument=None,
    verb=None,
    target=None,
    include_null=False
):

    tool = get_graph_tool()

    if query_type == "triplet":

        return tool.query_triplet(
            class_id=class_id,
            include_null=include_null
        )

    if query_type == "instrument":

        return tool.query_by_instrument(
            instrument=instrument,
            include_null=include_null
        )

    if query_type == "verb":

        return tool.query_by_verb(
            verb=verb,
            include_null=include_null
        )

    if query_type == "target":

        return tool.query_by_target(
            target=target,
            include_null=include_null
        )

    if query_type == "relation":

        return tool.query_relation(
            instrument=instrument,
            verb=verb,
            target=target,
            include_null=include_null
        )

    if query_type == "supported_actions":

        return tool.query_supported_actions(
            instrument=instrument
        )

    if query_type == "summary":

        return {
            "found": True,
            "summary": tool.get_graph_summary()
        }

    return {
        "found": False,
        "error": (
            f"不支持的 query_type: "
            f"{query_type}"
        )
    }
