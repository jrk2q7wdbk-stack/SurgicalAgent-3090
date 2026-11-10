import copy
import os
from typing import Optional


class ConversationMemory:
    """
    Surgical Agent 会话记忆。

    当前版本保存两类信息：

    1. 对话历史
       user / assistant

    2. 当前视觉上下文
       image_path
       best_triplet
       instruments
       verbs
       targets

    该 Memory 只存在于当前 Python 进程中。
    程序关闭后不会永久保存。
    """

    def __init__(
        self,
        max_history_turns: int = 10
    ):
        self.max_history_turns = max_history_turns

        # 对话历史
        self.history = []

        # 当前图像
        self.current_image_path: Optional[str] = None

        # 最近一次视觉模型完整输出
        self.last_vision_result = None

        # 最近一次知识查询结果
        self.last_knowledge_result = None

    # ========================================================
    # Conversation history
    # ========================================================

    def add_user_message(
        self,
        content: str
    ):
        self.history.append(
            {
                "role": "user",
                "content": content
            }
        )

        self._trim_history()

    def add_assistant_message(
        self,
        content: str
    ):
        self.history.append(
            {
                "role": "assistant",
                "content": content
            }
        )

        self._trim_history()

    def _trim_history(self):
        """
        max_history_turns=10
        表示最多保留约 10 轮 user-assistant 对话。
        """

        max_messages = (
            self.max_history_turns * 2
        )

        if len(self.history) > max_messages:
            self.history = self.history[
                -max_messages:
            ]

    def get_history(self):
        """
        返回副本，防止外部直接修改 memory。
        """

        return copy.deepcopy(
            self.history
        )

    def get_relevant_history(
        self,
        question: str,
        max_messages: int = 2
    ):
        """Return only conversation turns needed to resolve a reference.

        Direct visual and temporal questions should use structured memories,
        not the entire chat transcript.  A small tail is included only when
        the question contains an explicit follow-up/reference expression.
        """

        text = (question or "").lower()
        reference_markers = (
            "这个",
            "该",
            "它",
            "上述",
            "刚才",
            "前面",
            "上一",
            "then",
            "that",
            "it ",
        )

        if not any(marker in text for marker in reference_markers):
            return []

        safe_limit = max(0, min(int(max_messages), 4))
        return copy.deepcopy(self.history[-safe_limit:])

    # ========================================================
    # Vision memory
    # ========================================================

    def update_vision(
        self,
        image_path: str,
        vision_result: dict
    ):
        """
        更新当前视觉场景。
        """

        if image_path is not None:
            image_path = os.path.abspath(
                image_path
            )

        self.current_image_path = image_path

        self.last_vision_result = (
            copy.deepcopy(
                vision_result
            )
        )

    def get_best_triplet(self):
        """
        获取最近视觉结果中的 best_triplet。
        """

        if not self.last_vision_result:
            return None

        return self.last_vision_result.get(
            "best_triplet"
        )

    # ========================================================
    # Knowledge memory
    # ========================================================

    def update_knowledge(
        self,
        knowledge_result: dict
    ):
        self.last_knowledge_result = (
            copy.deepcopy(
                knowledge_result
            )
        )

    # ========================================================
    # Context generation
    # ========================================================

    def build_context_text(self) -> str:
        """
        将 Memory 转换成可提供给 LLM 的文本上下文。

        注意：
        这里只提供已有信息，
        不让 LLM 将 memory 当作新的视觉检测结果。
        """

        sections = []

        # ----------------------------------------------------
        # Vision context
        # ----------------------------------------------------

        if self.last_vision_result:

            best = (
                self.last_vision_result.get(
                    "best_triplet"
                )
            )

            if best:

                sections.append(
                    "独立静态图像识别结果（不是视频最新帧）：\n"
                    f"- instrument: "
                    f"{best.get('instrument')}\n"
                    f"- verb: "
                    f"{best.get('verb')}\n"
                    f"- target: "
                    f"{best.get('target')}\n"
                    f"- confidence: "
                    f"{best.get('confidence')}\n"
                    f"- confidence_level: "
                    f"{best.get('confidence_level')}\n"
                    f"- status: "
                    f"{best.get('status')}\n"
                    f"- answer_policy: "
                    f"{best.get('answer_policy')}"
                )

            if self.current_image_path:

                sections.append(
                    "静态图像来源（仅适用于此图片）：\n"
                    f"{self.current_image_path}"
                )

        # ----------------------------------------------------
        # Knowledge context
        # ----------------------------------------------------

        if self.last_knowledge_result:

            sections.append(
                "最近一次知识工具查询结果仍可作为"
                "当前对话的参考。"
            )

        if not sections:
            return "当前没有可用的历史视觉上下文。"

        return "\n\n".join(
            sections
        )

    # ========================================================
    # Reset
    # ========================================================

    def clear_conversation(self):
        """
        只清除聊天记录。
        保留当前视觉场景。
        """

        self.history.clear()

    def clear_visual_context(self):
        """
        清除当前视觉场景和知识上下文。
        """

        self.current_image_path = None
        self.last_vision_result = None
        self.last_knowledge_result = None

    def clear_all(self):
        """
        清空全部 Memory。
        """

        self.history.clear()

        self.current_image_path = None
        self.last_vision_result = None
        self.last_knowledge_result = None
