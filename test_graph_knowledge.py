from tools.graph_knowledge_tool import (
    query_surgical_graph
)


print("=" * 70)
print("TEST 1: IVT 58")
print("=" * 70)

result = query_surgical_graph(
    query_type="triplet",
    class_id=58
)

print(result)


print("\n" + "=" * 70)
print("TEST 2: hook")
print("=" * 70)

result = query_surgical_graph(
    query_type="instrument",
    instrument="hook"
)

print("count:", result["count"])

for triplet in result["triplets"]:
    print(
        triplet["class_id"],
        triplet["name"]
    )


print("\n" + "=" * 70)
print("TEST 3: cystic_duct")
print("=" * 70)

result = query_surgical_graph(
    query_type="target",
    target="cystic_duct"
)

print("count:", result["count"])

for triplet in result["triplets"]:
    print(
        triplet["class_id"],
        triplet["name"]
    )


print("\n" + "=" * 70)
print("TEST 4: hook + cystic_duct")
print("=" * 70)

result = query_surgical_graph(
    query_type="relation",
    instrument="hook",
    target="cystic_duct"
)

print("count:", result["count"])

for triplet in result["triplets"]:
    print(
        triplet["class_id"],
        triplet["name"]
    )


print("\n" + "=" * 70)
print("TEST 5: hook supported actions")
print("=" * 70)

result = query_surgical_graph(
    query_type="supported_actions",
    instrument="hook"
)

for action in result["actions"]:
    print(
        action["class_id"],
        action["name"]
    )


print("\n" + "=" * 70)
print("TEST 6: null triplet")
print("=" * 70)

result = query_surgical_graph(
    query_type="triplet",
    class_id=96
)

print(result)


print("\n" + "=" * 70)
print("TEST 7: graph summary")
print("=" * 70)

result = query_surgical_graph(
    query_type="summary"
)

print(result)