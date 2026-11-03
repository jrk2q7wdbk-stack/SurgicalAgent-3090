from collections import deque
from copy import deepcopy

from agent.confidence import classify_confidence, response_policy


class TemporalMemory:

    def __init__(
        self,
        max_events=100,
        confidence_threshold=0.3,
        switch_confirm_count=2
    ):
        """
        Temporal Memory

        功能：
        1. 每个采样帧只使用 best_triplet
        2. 低置信度 / null triplet 记为 no_action
        3. 连续出现 switch_confirm_count 次新状态后，
           才正式确认事件切换
        4. 单次瞬时异常自动忽略，避免时间线抖动

        参数
        ----------
        max_events : int
            最多保存多少个历史事件。

        confidence_threshold : float
            有效 triplet 的最低置信度。

        switch_confirm_count : int
            新状态连续出现多少次才确认切换。
            默认 2。
        """

        self.max_events = max_events

        self.confidence_threshold = (
            confidence_threshold
        )

        self.switch_confirm_count = (
            switch_confirm_count
        )

        # ====================================================
        # 已确认事件
        # ====================================================

        self.events = deque(
            maxlen=max_events
        )

        # ====================================================
        # 待确认状态
        # ====================================================

        self.pending_event = None

        self.pending_count = 0

    # ========================================================
    # 判断 best_triplet 是否有效
    # ========================================================

    def _is_valid_triplet(
        self,
        best_triplet
    ):

        if not best_triplet:
            return False

        confidence = float(
            best_triplet.get(
                "confidence",
                0.0
            )
        )

        # ----------------------------------------------------
        # 置信度不足
        # ----------------------------------------------------

        if (
            confidence
            <
            self.confidence_threshold
        ):
            return False

        # ----------------------------------------------------
        # postprocessing 已经明确标记低置信
        # ----------------------------------------------------

        status = best_triplet.get(
            "status"
        )

        if (
            status
            ==
            "low_confidence"
        ):
            return False

        instrument = (
            best_triplet.get(
                "instrument"
            )
        )

        verb = (
            best_triplet.get(
                "verb"
            )
        )

        target = (
            best_triplet.get(
                "target"
            )
        )

        # ----------------------------------------------------
        # null instrument
        # ----------------------------------------------------

        if instrument in (
            None,
            "",
            "null_instrument"
        ):
            return False

        # ----------------------------------------------------
        # null verb
        # ----------------------------------------------------

        if verb in (
            None,
            "",
            "null_verb"
        ):
            return False

        # ----------------------------------------------------
        # null target
        # ----------------------------------------------------

        if target in (
            None,
            "",
            "null_target"
        ):
            return False

        return True

    # ========================================================
    # Vision prediction → temporal label
    # ========================================================

    def _prediction_to_label(
        self,
        vision_result
    ):

        if not vision_result:

            return {
                "type":
                    "no_action"
            }

        best_triplet = (
            vision_result.get(
                "best_triplet"
            )
        )

        if best_triplet:
            instrument = best_triplet.get("instrument")
            verb = best_triplet.get("verb")
            target = best_triplet.get("target")
            complete = (
                instrument not in (None, "", "null_instrument")
                and verb not in (None, "", "null_verb")
                and target not in (None, "", "null_target")
            )
            confidence = float(best_triplet.get("confidence", 0.0))
            level = classify_confidence(confidence)

            # Low confidence is not equivalent to no action.  Preserve the
            # candidate as uncertain evidence so answers can explicitly refuse
            # a definite judgement instead of making a false no-action claim.
            if complete and level == "low":
                return {
                    "type": "uncertain",
                    "instrument": instrument,
                    "verb": verb,
                    "target": target,
                    "confidence": confidence,
                    "confidence_level": level,
                    "answer_policy": response_policy(level),
                }

        # ----------------------------------------------------
        # 没有可靠操作
        # ----------------------------------------------------

        if not self._is_valid_triplet(
            best_triplet
        ):

            return {
                "type":
                    "no_action"
            }

        # ----------------------------------------------------
        # 正常操作
        # ----------------------------------------------------

        return {
            "type":
                "action",

            "instrument":
                best_triplet.get(
                    "instrument"
                ),

            "verb":
                best_triplet.get(
                    "verb"
                ),

            "target":
                best_triplet.get(
                    "target"
                ),

            "confidence":
                float(
                    best_triplet.get(
                        "confidence",
                        0.0
                    )
                ),

            "confidence_level":
                classify_confidence(
                    best_triplet.get("confidence", 0.0)
                ),

            "answer_policy":
                response_policy(
                    classify_confidence(
                        best_triplet.get("confidence", 0.0)
                    )
                )
        }

    # ========================================================
    # 判断两个 label 是否相同
    # ========================================================

    def _same_label(
        self,
        label_a,
        label_b
    ):

        if (
            label_a is None
            or label_b is None
        ):
            return False

        if (
            label_a.get(
                "type"
            )
            !=
            label_b.get(
                "type"
            )
        ):
            return False

        # ----------------------------------------------------
        # 两个都是 no_action
        # ----------------------------------------------------

        if (
            label_a.get(
                "type"
            )
            ==
            "no_action"
        ):
            return True

        # ----------------------------------------------------
        # action 必须三元组完全一致
        # ----------------------------------------------------

        return (
            label_a.get(
                "instrument"
            )
            ==
            label_b.get(
                "instrument"
            )
            and
            label_a.get(
                "verb"
            )
            ==
            label_b.get(
                "verb"
            )
            and
            label_a.get(
                "target"
            )
            ==
            label_b.get(
                "target"
            )
        )

    # ========================================================
    # 获取当前正式事件对应 label
    # ========================================================

    def _get_current_label(
        self
    ):

        if not self.events:
            return None

        event = self.events[-1]

        event_type = event.get("type")

        if event_type == "no_action":

            return {
                "type":
                    "no_action"
            }

        return {
            "type":
                event_type,

            "instrument":
                event.get(
                    "instrument"
                ),

            "verb":
                event.get(
                    "verb"
                ),

            "target":
                event.get(
                    "target"
                )
        }

    # ========================================================
    # 创建正式事件
    # ========================================================

    def _create_event(
        self,
        timestamp,
        frame_id,
        label
    ):

        # ----------------------------------------------------
        # no_action
        # ----------------------------------------------------

        if (
            label.get(
                "type"
            )
            ==
            "no_action"
        ):

            event = {
                "type":
                    "no_action",

                "start_time":
                    timestamp,

                "end_time":
                    timestamp,

                "start_frame":
                    frame_id,

                "end_frame":
                    frame_id,

                "sample_points": [
                    {
                        "timestamp": timestamp,
                        "frame_id": frame_id
                    }
                ]
            }

            self.events.append(
                event
            )

            return

        # ----------------------------------------------------
        # action / uncertain candidate
        # ----------------------------------------------------

        confidence = float(
            label.get(
                "confidence",
                0.0
            )
        )

        event = {
            "type":
                label.get("type", "action"),

            "start_time":
                timestamp,

            "end_time":
                timestamp,

            "start_frame":
                frame_id,

            "end_frame":
                frame_id,

            "instrument":
                label.get(
                    "instrument"
                ),

            "verb":
                label.get(
                    "verb"
                ),

            "target":
                label.get(
                    "target"
                ),

            "confidences": [
                confidence
            ],

            "sample_points": [
                {
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "confidence": confidence,
                    "confidence_level": classify_confidence(confidence)
                }
            ],

            "mean_confidence":
                confidence,

            "confidence_level":
                classify_confidence(confidence),

            "answer_policy":
                response_policy(
                    classify_confidence(confidence)
                )
        }

        self.events.append(
            event
        )

    # ========================================================
    # 扩展当前正式事件
    # ========================================================

    def _extend_current_event(
        self,
        timestamp,
        frame_id,
        label
    ):

        if not self.events:
            return

        event = self.events[-1]

        event[
            "end_time"
        ] = timestamp

        event[
            "end_frame"
        ] = frame_id

        event.setdefault("sample_points", []).append(
            {
                "timestamp": timestamp,
                "frame_id": frame_id,
                **(
                    {
                        "confidence": float(label.get("confidence", 0.0)),
                        "confidence_level": classify_confidence(
                            label.get("confidence", 0.0)
                        )
                    }
                    if event.get("type") != "no_action"
                    else {}
                )
            }
        )

        # ----------------------------------------------------
        # no_action 没有 confidence
        # ----------------------------------------------------

        if event.get("type") == "no_action":
            return

        confidence = float(
            label.get(
                "confidence",
                0.0
            )
        )

        event[
            "confidences"
        ].append(
            confidence
        )

        event[
            "mean_confidence"
        ] = (
            sum(
                event[
                    "confidences"
                ]
            )
            /
            len(
                event[
                    "confidences"
                ]
            )
        )

        event["confidence_level"] = classify_confidence(
            event["mean_confidence"]
        )
        event["answer_policy"] = response_policy(
            event["confidence_level"]
        )

    # ========================================================
    # 设置 pending
    # ========================================================

    def _set_pending(
        self,
        timestamp,
        frame_id,
        label
    ):

        self.pending_event = {
            "label":
                deepcopy(
                    label
                ),

            "start_time":
                timestamp,

            "end_time":
                timestamp,

            "start_frame":
                frame_id,

            "end_frame":
                frame_id,

            "samples": [
                deepcopy(
                    label
                )
            ],

            "sample_points": [
                {
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    **(
                        {
                            "confidence": float(label.get("confidence", 0.0)),
                            "confidence_level": classify_confidence(
                                label.get("confidence", 0.0)
                            )
                        }
                        if label.get("type") != "no_action"
                        else {}
                    )
                }
            ]
        }

        self.pending_count = 1

    # ========================================================
    # 扩展 pending
    # ========================================================

    def _extend_pending(
        self,
        timestamp,
        frame_id,
        label
    ):

        if not self.pending_event:
            return

        self.pending_event[
            "end_time"
        ] = timestamp

        self.pending_event[
            "end_frame"
        ] = frame_id

        self.pending_event[
            "samples"
        ].append(
            deepcopy(
                label
            )
        )

        self.pending_event.setdefault("sample_points", []).append(
            {
                "timestamp": timestamp,
                "frame_id": frame_id,
                **(
                    {
                        "confidence": float(label.get("confidence", 0.0)),
                        "confidence_level": classify_confidence(
                            label.get("confidence", 0.0)
                        )
                    }
                    if label.get("type") != "no_action"
                    else {}
                )
            }
        )

        self.pending_count += 1

    # ========================================================
    # 清空 pending
    # ========================================================

    def _clear_pending(
        self
    ):

        self.pending_event = None

        self.pending_count = 0

    # ========================================================
    # 确认 pending → 正式事件
    # ========================================================

    def _confirm_pending(
        self
    ):

        if not self.pending_event:
            return

        pending = (
            self.pending_event
        )

        label = (
            pending[
                "label"
            ]
        )

        # ====================================================
        # no_action
        # ====================================================

        if (
            label.get(
                "type"
            )
            ==
            "no_action"
        ):

            event = {
                "type":
                    "no_action",

                "start_time":
                    pending[
                        "start_time"
                    ],

                "end_time":
                    pending[
                        "end_time"
                    ],

                "start_frame":
                    pending[
                        "start_frame"
                    ],

                "end_frame":
                    pending[
                        "end_frame"
                    ],

                "sample_points":
                    deepcopy(
                        pending.get("sample_points", [])
                    )
            }

            self.events.append(
                event
            )

            self._clear_pending()

            return

        # ====================================================
        # action / uncertain candidate
        # ====================================================

        confidences = []

        for sample in pending[
            "samples"
        ]:

            confidence = float(
                sample.get(
                    "confidence",
                    0.0
                )
            )

            confidences.append(
                confidence
            )

        mean_confidence = (
            sum(
                confidences
            )
            /
            len(
                confidences
            )
            if confidences
            else 0.0
        )

        event = {
            "type":
                label.get("type", "action"),

            "start_time":
                pending[
                    "start_time"
                ],

            "end_time":
                pending[
                    "end_time"
                ],

            "start_frame":
                pending[
                    "start_frame"
                ],

            "end_frame":
                pending[
                    "end_frame"
                ],

            "instrument":
                label.get(
                    "instrument"
                ),

            "verb":
                label.get(
                    "verb"
                ),

            "target":
                label.get(
                    "target"
                ),

            "confidences":
                confidences,

            "sample_points":
                deepcopy(
                    pending.get("sample_points", [])
                ),

            "mean_confidence":
                mean_confidence,

            "confidence_level":
                classify_confidence(mean_confidence),

            "answer_policy":
                response_policy(
                    classify_confidence(mean_confidence)
                )
        }

        self.events.append(
            event
        )

        self._clear_pending()

    # ========================================================
    # 添加一个采样帧预测
    # ========================================================

    def add_prediction(
        self,
        timestamp,
        frame_id,
        vision_result
    ):

        new_label = (
            self._prediction_to_label(
                vision_result
            )
        )

        # ====================================================
        # 第一帧
        # ====================================================

        if not self.events:

            self._create_event(
                timestamp=timestamp,
                frame_id=frame_id,
                label=new_label
            )

            return

        current_label = (
            self._get_current_label()
        )

        # ====================================================
        # 情况 1：
        # 当前预测与已确认状态相同
        #
        # A A A
        #
        # 或：
        #
        # A A B A
        #
        # B 只出现一次，被认为是抖动
        # ====================================================

        if self._same_label(
            new_label,
            current_label
        ):

            # ------------------------------------------------
            # 如果之前存在 pending，
            # 说明新状态没有持续出现
            # ------------------------------------------------

            if self.pending_event:

                self._clear_pending()

            self._extend_current_event(
                timestamp=timestamp,
                frame_id=frame_id,
                label=new_label
            )

            return

        # ====================================================
        # 情况 2：
        # 出现一个潜在的新状态
        # ====================================================

        if not self.pending_event:

            self._set_pending(
                timestamp=timestamp,
                frame_id=frame_id,
                label=new_label
            )

            if self.pending_count >= self.switch_confirm_count:
                self._confirm_pending()

            return

        pending_label = (
            self.pending_event[
                "label"
            ]
        )

        # ====================================================
        # 情况 3：
        # 新状态连续出现
        #
        # A A B B
        #
        # 当 B 连续达到 switch_confirm_count，
        # 才正式切换
        # ====================================================

        if self._same_label(
            new_label,
            pending_label
        ):

            self._extend_pending(
                timestamp=timestamp,
                frame_id=frame_id,
                label=new_label
            )

            if (
                self.pending_count
                >=
                self.switch_confirm_count
            ):

                self._confirm_pending()

            return

        # ====================================================
        # 情况 4：
        # pending 期间又出现第三种状态
        #
        # A A B C
        #
        # B 不够稳定，丢弃
        # C 成为新的 pending
        # ====================================================

        # Epistemic states must not disappear from the timeline.  A single
        # no_action/uncertain sample is meaningful evidence, even though a
        # single competing action label can still be treated as jitter.
        if pending_label.get("type") in ("no_action", "uncertain"):
            self._confirm_pending()

        self._set_pending(
            timestamp=timestamp,
            frame_id=frame_id,
            label=new_label
        )

        if self.pending_count >= self.switch_confirm_count:
            self._confirm_pending()

    # ========================================================
    # 获取全部事件
    # ========================================================

    def get_events(
        self
    ):

        return deepcopy(
            list(
                self.events
            )
        )

    # ========================================================
    # 获取最近 N 个事件
    # ========================================================

    def get_recent_events(
        self,
        count=5
    ):

        events = list(
            self.events
        )

        return deepcopy(
            events[
                -count:
            ]
        )

    # ========================================================
    # 获取最新事件
    # ========================================================

    def get_latest_event(
        self
    ):

        if not self.events:
            return None

        return deepcopy(
            self.events[-1]
        )

    # ========================================================
    # 构建给 LLM 使用的时间上下文
    # ========================================================

    def build_context_text(
        self,
        count=10
    ):

        events = (
            self.get_recent_events(
                count=count
            )
        )

        if not events:

            return (
                "当前没有可用的时间序列手术事件。"
            )

        lines = [
            "最近手术事件："
        ]

        for index, event in enumerate(
            events,
            start=1
        ):

            start_time = float(
                event.get(
                    "start_time",
                    0.0
                )
            )

            end_time = float(
                event.get(
                    "end_time",
                    start_time
                )
            )

            # =================================================
            # no_action
            # =================================================

            if (
                event.get(
                    "type"
                )
                ==
                "no_action"
            ):

                lines.append(
                    f"{index}. "
                    f"{start_time:.1f}s - "
                    f"{end_time:.1f}s："
                    f"no_action"
                    f"（未检测到明确器械操作）"
                )

                continue

            # =================================================
            # uncertain candidate
            # =================================================

            if event.get("type") == "uncertain":

                instrument = event.get("instrument", "unknown")
                verb = event.get("verb", "unknown")
                target = event.get("target", "unknown")
                mean_confidence = float(event.get("mean_confidence", 0.0))

                lines.append(
                    f"{index}. "
                    f"{start_time:.1f}s - {end_time:.1f}s："
                    f"uncertain（最高候选 {instrument} - {verb} - {target}，"
                    f"平均置信度 {mean_confidence:.3f}；不能作确定判断）"
                )
                continue

            # =================================================
            # action
            # =================================================

            instrument = (
                event.get(
                    "instrument",
                    "unknown"
                )
            )

            verb = (
                event.get(
                    "verb",
                    "unknown"
                )
            )

            target = (
                event.get(
                    "target",
                    "unknown"
                )
            )

            mean_confidence = float(
                event.get(
                    "mean_confidence",
                    0.0
                )
            )

            lines.append(
                f"{index}. "
                f"{start_time:.1f}s - "
                f"{end_time:.1f}s："
                f"{instrument} - "
                f"{verb} - "
                f"{target}，"
                f"平均置信度 "
                f"{mean_confidence:.3f}"
            )

        return "\n".join(
            lines
        )

    # ========================================================
    # 查询指定时间范围内的事件
    # ========================================================

    def get_events_by_time(
        self,
        start_time=None,
        end_time=None
    ):
        """
        查询指定时间范围内发生的事件。

        只要事件和查询区间有重叠，就返回。
        """

        events = list(
            self.events
        )

        if not events:
            return []

        if start_time is None:
            start_time = float("-inf")

        if end_time is None:
            end_time = float("inf")

        results = []

        for event in events:

            event_start = float(
                event.get(
                    "start_time",
                    0.0
                )
            )

            event_end = float(
                event.get(
                    "end_time",
                    event_start
                )
            )

            # 没有时间交集
            if (
                event_end < start_time
                or
                event_start > end_time
            ):
                continue

            results.append(
                deepcopy(event)
            )

        # For an exact time-point query, expose the nearest sampled frame's
        # confidence as well as the event aggregate.  This prevents a high
        # event mean from masking a medium-confidence boundary frame.
        if (
            start_time != float("-inf")
            and end_time != float("inf")
            and float(start_time) == float(end_time)
        ):
            query_time = float(start_time)
            for result in results:
                points = result.get("sample_points") or []
                if not points:
                    continue
                nearest = min(
                    points,
                    key=lambda point: abs(
                        float(point.get("timestamp", query_time)) - query_time
                    )
                )
                result["query_sample"] = deepcopy(nearest)
                if "confidence" in nearest:
                    level = classify_confidence(nearest["confidence"])
                    result["query_confidence"] = float(nearest["confidence"])
                    result["query_confidence_level"] = level
                    result["query_answer_policy"] = response_policy(level)

        return results

    # ========================================================
    # 获取时序统计信息
    # ========================================================

    def get_statistics(
        self
    ):
        """
        返回当前视频时序事件统计。
        """

        events = list(
            self.events
        )

        action_events = [
            event
            for event in events
            if event.get("type") == "action"
        ]

        no_action_events = [
            event
            for event in events
            if event.get("type") == "no_action"
        ]

        uncertain_events = [
            event
            for event in events
            if event.get("type") == "uncertain"
        ]

        action_duration = 0.0
        no_action_duration = 0.0

        for event in action_events:

            start_time = float(
                event.get(
                    "start_time",
                    0.0
                )
            )

            end_time = float(
                event.get(
                    "end_time",
                    start_time
                )
            )

            action_duration += max(
                0.0,
                end_time - start_time
            )

        for event in no_action_events:

            start_time = float(
                event.get(
                    "start_time",
                    0.0
                )
            )

            end_time = float(
                event.get(
                    "end_time",
                    start_time
                )
            )

            no_action_duration += max(
                0.0,
                end_time - start_time
            )

        return {
            "total_events":
                len(events),

            "action_events":
                len(action_events),

            "no_action_events":
                len(no_action_events),

            "uncertain_events":
                len(uncertain_events),

            "action_duration":
                action_duration,

            "no_action_duration":
                no_action_duration
        }

    # ========================================================
    # 获取持续时间最长的明确操作
    # ========================================================

    def get_longest_action(
        self
    ):
        """
        返回持续时间最长的 action 事件。
        """

        longest_event = None
        longest_duration = -1.0

        for event in self.events:

            if (
                event.get(
                    "type"
                )
                !=
                "action"
            ):
                continue

            start_time = float(
                event.get(
                    "start_time",
                    0.0
                )
            )

            end_time = float(
                event.get(
                    "end_time",
                    start_time
                )
            )

            duration = max(
                0.0,
                end_time - start_time
            )

            if (
                duration
                >
                longest_duration
            ):

                longest_duration = duration

                longest_event = deepcopy(
                    event
                )

        if longest_event is None:
            return None

        longest_event[
            "duration"
        ] = longest_duration

        return longest_event

    # ========================================================
    # 获取操作转换序列
    # ========================================================

    def get_transitions(
        self
    ):
        """
        返回相邻事件之间的变化。
        """

        events = list(
            self.events
        )

        if len(events) < 2:
            return []

        transitions = []

        for index in range(
            len(events) - 1
        ):

            current_event = (
                deepcopy(
                    events[index]
                )
            )

            next_event = (
                deepcopy(
                    events[index + 1]
                )
            )

            transitions.append(
                {
                    "from":
                        current_event,

                    "to":
                        next_event
                }
            )

        return transitions

    def finalize_pending(self):
        """Close the stream without silently dropping its final sample.

        A pending state that has enough samples is confirmed normally.  A
        single trailing low-confidence sample is preserved as an ``uncertain``
        event; other one-frame switches remain suppressed as temporal jitter.
        """

        if not self.pending_event:
            return

        label = self.pending_event.get("label", {})
        if self.pending_count >= self.switch_confirm_count:
            self._confirm_pending()
        elif label.get("type") == "uncertain":
            self._confirm_pending()
        else:
            self._clear_pending()

    # ========================================================
    # 清空所有 Temporal Memory
    # ========================================================


    def clear(
        self
    ):

        self.events.clear()

        self._clear_pending()
