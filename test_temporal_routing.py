from agent.query_router import route_temporal_question
from agent.temporal_memory import TemporalMemory


def prediction(confidence, instrument="hook", verb="dissect", target="cystic_duct"):
    return {
        "best_triplet": {
            "instrument": instrument,
            "verb": verb,
            "target": target,
            "confidence": confidence,
            "status": (
                "low_confidence" if confidence < 0.5 else "high_confidence"
            ),
        }
    }


def build_memory():
    memory = TemporalMemory(switch_confirm_count=1)
    memory.add_prediction(0.0, 0, prediction(0.95))
    memory.add_prediction(
        10.0, 10, prediction(0.90, "grasper", "grasp", "gallbladder")
    )
    memory.add_prediction(
        20.0, 20, prediction(0.85, "clipper", "clip", "cystic_duct")
    )
    return memory


def test_low_confidence_is_uncertain_not_no_action():
    memory = TemporalMemory()
    memory.add_prediction(0.0, 0, prediction(0.30))
    event = memory.get_latest_event()
    assert event["type"] == "uncertain"
    assert event["answer_policy"] == "refuse_definite_judgment"


def test_null_label_is_no_action():
    memory = TemporalMemory()
    memory.add_prediction(
        0.0,
        0,
        prediction(0.95, instrument="grasper", verb="null_verb", target="gallbladder"),
    )
    assert memory.get_latest_event()["type"] == "no_action"


def test_single_trailing_uncertain_sample_is_not_dropped():
    memory = TemporalMemory(switch_confirm_count=2)
    memory.add_prediction(0.0, 0, prediction(0.95))
    memory.add_prediction(
        2.0, 2, prediction(0.30, "grasper", "grasp", "gallbladder")
    )
    memory.finalize_pending()
    assert [item["type"] for item in memory.get_events()] == ["action", "uncertain"]


def test_explicit_range_routes_to_only_that_range():
    route = route_temporal_question("20秒到35秒之间发生了什么？", build_memory())
    assert route == {
        "intent": "time_range",
        "query_type": "time_range",
        "start_time": 20.0,
        "end_time": 35.0,
    }


def test_recent_ten_seconds_uses_latest_event_time():
    route = route_temporal_question("最近十秒发生了什么？", build_memory())
    assert route["query_type"] == "time_range"
    assert route["start_time"] == 10.0
    assert route["end_time"] == 20.0


def test_previous_event_routes_to_two_recent_events():
    route = route_temporal_question("上一个事件是什么？", build_memory())
    assert route["query_type"] == "recent"
    assert route["count"] == 2
    assert route["selection"] == "previous"


def test_point_question_routes_to_a_point_interval():
    route = route_temporal_question("20秒时模型识别到了什么？", build_memory())
    assert route["start_time"] == 20.0
    assert route["end_time"] == 20.0


def test_point_query_uses_sample_confidence_not_only_event_mean():
    memory = TemporalMemory()
    memory.add_prediction(0.0, 0, prediction(0.60))
    memory.add_prediction(2.0, 2, prediction(0.95))
    memory.add_prediction(4.0, 4, prediction(0.95))

    event = memory.get_events_by_time(0.0, 0.0)[0]
    assert event["mean_confidence"] > 0.75
    assert event["query_confidence"] == 0.60
    assert event["query_confidence_level"] == "medium"
    assert event["query_answer_policy"] == "conservative"


def test_single_no_action_sample_is_preserved_before_a_third_state():
    memory = TemporalMemory(switch_confirm_count=2)
    memory.add_prediction(0.0, 0, prediction(0.95))
    memory.add_prediction(
        2.0,
        2,
        prediction(0.10, instrument="hook", verb="null_verb", target="null_target"),
    )
    memory.add_prediction(
        4.0, 4, prediction(0.30, "grasper", "grasp", "gallbladder")
    )
    memory.finalize_pending()
    assert [item["type"] for item in memory.get_events()] == [
        "action",
        "no_action",
        "uncertain",
    ]
