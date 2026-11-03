from agent.memory import ConversationMemory


def test_direct_question_does_not_receive_full_chat_history():
    memory = ConversationMemory(max_history_turns=10)
    memory.add_user_message("旧问题")
    memory.add_assistant_message("旧回答")
    assert memory.get_relevant_history("20秒发生了什么？") == []


def test_reference_question_receives_only_last_turn():
    memory = ConversationMemory(max_history_turns=10)
    for index in range(5):
        memory.add_user_message(f"问题{index}")
        memory.add_assistant_message(f"回答{index}")
    relevant = memory.get_relevant_history("这个操作是什么意思？")
    assert len(relevant) == 2
    assert relevant[0]["content"] == "问题4"
    assert relevant[1]["content"] == "回答4"
