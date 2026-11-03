from agent.llm_agent import SurgicalLLMAgent


agent = SurgicalLLMAgent()


print("=" * 70)
print("Q1")
print("=" * 70)

answer = agent.answer(
    question="请分析这段视频。",
    video_path="../test.mp4"
)

print(answer)


print("\n" + "=" * 70)
print("Q2")
print("=" * 70)

answer = agent.answer(
    question="20秒到35秒之间发生了什么？"
)

print(answer)


print("\n" + "=" * 70)
print("Q3")
print("=" * 70)

answer = agent.answer(
    question="哪个明确操作持续时间最长？"
)

print(answer)


print("\n" + "=" * 70)
print("Q4")
print("=" * 70)

answer = agent.answer(
    question="这段视频总共包含多少个操作阶段？"
)

print(answer)


print("\n" + "=" * 70)
print("Q5")
print("=" * 70)

answer = agent.answer(
    question="前后操作发生了哪些变化？"
)

print(answer)