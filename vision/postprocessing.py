import torch

from agent.confidence import enrich_triplet


def _decode_multilabel(probabilities, names, threshold):
    """将多标签概率解码为类别名称和置信度。"""
    results = []

    for idx, prob in enumerate(probabilities):
        if prob >= threshold:
            results.append({
                "id": idx,
                "name": names[idx],
                "confidence": float(prob)
            })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def _decode_triplets(probabilities, triplet_classes, threshold, top_k=None):
    """将 100 类 triplet 概率解码为 Instrument-Verb-Target。"""
    results = []

    for idx, prob in enumerate(probabilities):
        if prob < threshold:
            continue

        triplet = triplet_classes[idx]

        results.append(enrich_triplet({
            "id": idx,
            "instrument": triplet["instrument"],
            "verb": triplet["verb"],
            "target": triplet["target"],
            "confidence": float(prob)
        }))

    results.sort(key=lambda x: x["confidence"], reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results


def postprocess(
    logit_i,
    logit_v,
    logit_t,
    logit_ivt,
    instrument_names,
    verb_names,
    target_names,
    triplet_classes,
    threshold=0.5,
    triplet_threshold=0.5,
    top_k_triplets=5
):
    """
    对模型输出进行后处理，返回 instrument / verb / target / triplet。
    """

    prob_i = torch.sigmoid(logit_i).detach().cpu().numpy().flatten()
    prob_v = torch.sigmoid(logit_v).detach().cpu().numpy().flatten()
    prob_t = torch.sigmoid(logit_t).detach().cpu().numpy().flatten()
    prob_ivt = torch.sigmoid(logit_ivt).detach().cpu().numpy().flatten()

    instruments = _decode_multilabel(
        prob_i,
        instrument_names,
        threshold
    )

    verbs = _decode_multilabel(
        prob_v,
        verb_names,
        threshold
    )

    targets = _decode_multilabel(
        prob_t,
        target_names,
        threshold
    )

    triplets = _decode_triplets(
        prob_ivt,
        triplet_classes,
        triplet_threshold,
        top_k_triplets
    )

    # 若没有 triplet 超过阈值，仍返回最高概率候选，
    # 但标记为 low_confidence，方便后续 Agent 谨慎使用。
    if triplets:
        best_triplet = dict(triplets[0])
    elif len(prob_ivt) > 0:
        top_idx = int(prob_ivt.argmax())
        triplet = triplet_classes[top_idx]

        best_triplet = enrich_triplet({
            "id": top_idx,
            "instrument": triplet["instrument"],
            "verb": triplet["verb"],
            "target": triplet["target"],
            "confidence": float(prob_ivt[top_idx])
        })
    else:
        best_triplet = None

    return {
        "instruments": instruments,
        "verbs": verbs,
        "targets": targets,
        "triplets": triplets,
        "best_triplet": best_triplet
    }
