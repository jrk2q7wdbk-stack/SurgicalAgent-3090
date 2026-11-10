import json
import os


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

KNOWLEDGE_DIR = os.path.join(
    PROJECT_ROOT,
    "knowledge"
)


INSTRUMENT_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "instrument_names.json"
)

VERB_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "verb_names.json"
)

TARGET_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "target_names.json"
)

TRIPLET_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "triplet_classes.json"
)

SURGICAL_KNOWLEDGE_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "surgical_knowledge.json"
)

# 如果你的 label_mapping.txt 不在 knowledge/
# 而是在项目根目录，把这里改成 PROJECT_ROOT。
LABEL_MAPPING_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "label_mapping.txt"
)

OUTPUT_PATH = os.path.join(
    KNOWLEDGE_DIR,
    "surgical_knowledge_graph.json"
)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def load_label_mapping(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            values = [
                int(x.strip())
                for x in line.split(",")
            ]

            if len(values) != 6:
                raise ValueError(
                    f"Invalid mapping row: {line}"
                )

            rows.append({
                "ivt_id": values[0],
                "instrument_id": values[1],
                "verb_id": values[2],
                "target_id": values[3],
                "iv_id": values[4],
                "it_id": values[5]
            })

    return rows


def build_graph():

    instruments = load_json(
        INSTRUMENT_PATH
    )

    verbs = load_json(
        VERB_PATH
    )

    targets = load_json(
        TARGET_PATH
    )

    triplet_classes = load_json(
        TRIPLET_PATH
    )

    surgical_knowledge = load_json(
        SURGICAL_KNOWLEDGE_PATH
    )

    mappings = load_label_mapping(
        LABEL_MAPPING_PATH
    )

    # ========================================================
    # Basic Validation
    # ========================================================

    assert len(instruments) == 6, (
        f"Expected 6 instruments, got {len(instruments)}"
    )

    assert len(verbs) == 10, (
        f"Expected 10 verbs, got {len(verbs)}"
    )

    assert len(targets) == 15, (
        f"Expected 15 targets, got {len(targets)}"
    )

    assert len(triplet_classes) == 100, (
        f"Expected 100 triplets, got {len(triplet_classes)}"
    )

    assert len(mappings) == 100, (
        f"Expected 100 mappings, got {len(mappings)}"
    )

    # ========================================================
    # Nodes
    # ========================================================

    nodes = []

    # --------------------------------------------------------
    # Instrument Nodes
    # --------------------------------------------------------

    for idx, name in enumerate(instruments):

        knowledge = (
            surgical_knowledge
            .get("instruments", {})
            .get(name, {})
        )

        nodes.append({
            "id": f"instrument:{idx}",
            "type": "instrument",
            "class_id": idx,
            "name": name,
            "description": knowledge.get(
                "description"
            ),
            "functions": knowledge.get(
                "functions",
                []
            )
        })

    # --------------------------------------------------------
    # Verb Nodes
    # --------------------------------------------------------

    for idx, name in enumerate(verbs):

        knowledge = (
            surgical_knowledge
            .get("verbs", {})
            .get(name, {})
        )

        nodes.append({
            "id": f"verb:{idx}",
            "type": "verb",
            "class_id": idx,
            "name": name,
            "description": knowledge.get(
                "description"
            )
        })

    # --------------------------------------------------------
    # Target Nodes
    # --------------------------------------------------------

    for idx, name in enumerate(targets):

        knowledge = (
            surgical_knowledge
            .get("targets", {})
            .get(name, {})
        )

        nodes.append({
            "id": f"target:{idx}",
            "type": "target",
            "class_id": idx,
            "name": name,
            "description": knowledge.get(
                "description"
            )
        })

    # --------------------------------------------------------
    # Triplet Nodes
    # --------------------------------------------------------

    for mapping in mappings:

        ivt_id = mapping["ivt_id"]

        triplet = triplet_classes[
            ivt_id
        ]

        instrument_id = mapping[
            "instrument_id"
        ]

        verb_id = mapping[
            "verb_id"
        ]

        target_id = mapping[
            "target_id"
        ]

        # ====================================================
        # Critical consistency check
        # ====================================================

        expected_instrument = instruments[
            instrument_id
        ]

        expected_verb = verbs[
            verb_id
        ]

        expected_target = targets[
            target_id
        ]

        actual_instrument = triplet[
            "instrument"
        ]

        actual_verb = triplet[
            "verb"
        ]

        actual_target = triplet[
            "target"
        ]

        if actual_instrument != expected_instrument:
            raise ValueError(
                f"IVT {ivt_id}: "
                f"instrument mismatch: "
                f"{actual_instrument} != "
                f"{expected_instrument}"
            )

        if actual_verb != expected_verb:
            raise ValueError(
                f"IVT {ivt_id}: "
                f"verb mismatch: "
                f"{actual_verb} != "
                f"{expected_verb}"
            )

        if actual_target != expected_target:
            raise ValueError(
                f"IVT {ivt_id}: "
                f"target mismatch: "
                f"{actual_target} != "
                f"{expected_target}"
            )

        is_null = (
            actual_verb == "null_verb"
            or actual_target == "null_target"
        )

        nodes.append({
            "id": f"triplet:{ivt_id}",
            "type": "triplet",
            "class_id": ivt_id,

            "name": (
                f"{actual_instrument}-"
                f"{actual_verb}-"
                f"{actual_target}"
            ),

            "instrument":
                actual_instrument,

            "verb":
                actual_verb,

            "target":
                actual_target,

            "instrument_id":
                instrument_id,

            "verb_id":
                verb_id,

            "target_id":
                target_id,

            "iv_id":
                mapping["iv_id"],

            "it_id":
                mapping["it_id"],

            "is_null":
                is_null
        })

    # ========================================================
    # Edges
    # ========================================================

    edges = []

    edge_counter = 0

    def add_edge(
        source,
        relation,
        target,
        metadata=None
    ):

        nonlocal edge_counter

        edge = {
            "id": f"edge:{edge_counter}",
            "source": source,
            "relation": relation,
            "target": target
        }

        if metadata:
            edge["metadata"] = metadata

        edges.append(edge)

        edge_counter += 1

    # --------------------------------------------------------
    # Triplet structural relations
    # --------------------------------------------------------

    for mapping in mappings:

        ivt_id = mapping["ivt_id"]

        triplet_node = (
            f"triplet:{ivt_id}"
        )

        instrument_node = (
            f"instrument:"
            f"{mapping['instrument_id']}"
        )

        verb_node = (
            f"verb:"
            f"{mapping['verb_id']}"
        )

        target_node = (
            f"target:"
            f"{mapping['target_id']}"
        )

        add_edge(
            triplet_node,
            "uses_instrument",
            instrument_node
        )

        add_edge(
            triplet_node,
            "has_action",
            verb_node
        )

        add_edge(
            triplet_node,
            "acts_on",
            target_node
        )

    # --------------------------------------------------------
    # Instrument → supported action
    # Based on current knowledge base
    # --------------------------------------------------------

    verb_to_id = {
        name: idx
        for idx, name in enumerate(verbs)
    }

    for instrument_id, instrument_name in enumerate(
        instruments
    ):

        knowledge = (
            surgical_knowledge
            .get("instruments", {})
            .get(instrument_name, {})
        )

        functions = knowledge.get(
            "functions",
            []
        )

        for function_name in functions:

            # Only create graph relation when this function
            # belongs to the CholecT50 verb vocabulary.
            if function_name not in verb_to_id:
                continue

            add_edge(
                f"instrument:{instrument_id}",
                "supports_action",
                f"verb:{verb_to_id[function_name]}",
                metadata={
                    "source":
                        "surgical_knowledge.json"
                }
            )

    # ========================================================
    # Graph
    # ========================================================

    graph = {

        "metadata": {
            "name":
                "SurgicalAgent Knowledge Graph",

            "version":
                "1.0",

            "dataset":
                "CholecT50",

            "num_instruments":
                len(instruments),

            "num_verbs":
                len(verbs),

            "num_targets":
                len(targets),

            "num_triplets":
                len(triplet_classes),

            "num_nodes":
                len(nodes),

            "num_edges":
                len(edges),

            "source_files": [
                "instrument_names.json",
                "verb_names.json",
                "target_names.json",
                "triplet_classes.json",
                "label_mapping.txt",
                "surgical_knowledge.json"
            ]
        },

        "nodes":
            nodes,

        "edges":
            edges
    }

    # ========================================================
    # Save
    # ========================================================

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            graph,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # Report
    # ========================================================

    print("=" * 70)
    print("Surgical Knowledge Graph Build")
    print("=" * 70)

    print(
        f"[PASS] instruments: "
        f"{len(instruments)}"
    )

    print(
        f"[PASS] verbs: "
        f"{len(verbs)}"
    )

    print(
        f"[PASS] targets: "
        f"{len(targets)}"
    )

    print(
        f"[PASS] triplets: "
        f"{len(triplet_classes)}"
    )

    print(
        f"[PASS] mapping rows: "
        f"{len(mappings)}"
    )

    print(
        "[PASS] triplet_classes "
        "matches label_mapping"
    )

    print(
        f"[PASS] graph nodes: "
        f"{len(nodes)}"
    )

    print(
        f"[PASS] graph edges: "
        f"{len(edges)}"
    )

    print(
        f"[PASS] saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    build_graph()