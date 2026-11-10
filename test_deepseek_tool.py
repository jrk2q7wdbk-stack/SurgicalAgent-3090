import os
import json
import sys
from openai import OpenAI


# ============================================================
# 1. 读取 DeepSeek API Key
# ============================================================

api_key = os.getenv("DEEPSEEK_API_KEY")

if "pytest" in sys.modules and os.getenv("RUN_DEEPSEEK_API_TESTS") != "1":
    import pytest
    pytest.skip("external paid DeepSeek test; set RUN_DEEPSEEK_API_TESTS=1", allow_module_level=True)

if not api_key:
    raise ValueError(
        "没有检测到 DEEPSEEK_API_KEY，请先配置 DeepSeek API Key。"
    )


# ============================================================
# 2. 创建 DeepSeek Client
# ============================================================

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# ============================================================
# 3. 定义一个最简单的测试工具
# ============================================================

TOOLS = [
    {
        "type": "function",
        "name": "query_surgical_action",
        "description": "查询手术视频中某个时间点正在进行的手术操作。",
        "parameters": {
            "type": "object",
            "properties": {
                "time_seconds": {
                    "type": "number",
                    "description": "需要查询的视频时间点，单位为秒。"
                }
            },
            "required": ["time_seconds"],
            "additionalProperties": False
        }
    }
]


# ============================================================
# 4. 调用 DeepSeek
# ============================================================

print("=" * 60)
print("[Test] DeepSeek Tool Calling 测试")
print("=" * 60)

question = "请查询手术视频第20秒正在进行什么操作。"

print(f"\n[Test] 用户问题：{question}")
print("\n[Test] 正在请求 DeepSeek...")


try:
    response = client.responses.create(
        model="deepseek-v4-flash",
        input=question,
        tools=TOOLS
    )

    print("\n[Test] DeepSeek 返回成功。")

    # ========================================================
    # 5. 检查 DeepSeek 是否产生 Tool Call
    # ========================================================

    tool_called = False

    for item in response.output:

        print(
            f"[Test] Output item type: "
            f"{getattr(item, 'type', 'unknown')}"
        )

        if item.type == "function_call":

            tool_called = True

            print("\n" + "=" * 60)
            print("Tool Calling 成功！")
            print("=" * 60)

            print(f"工具名称：{item.name}")
            print(f"原始参数：{item.arguments}")

            try:
                arguments = json.loads(item.arguments)

                print("\n解析后的参数：")
                print(
                    json.dumps(
                        arguments,
                        ensure_ascii=False,
                        indent=2
                    )
                )

            except json.JSONDecodeError:
                print(
                    "\n[Warning] Tool 参数不是合法 JSON。"
                )

    # ========================================================
    # 6. 判断最终测试结果
    # ========================================================

    if tool_called:

        print("\n" + "=" * 60)
        print("[PASS] DeepSeek Tool Calling 工作正常")
        print("=" * 60)

        print(
            "\n下一步可以把 DeepSeek 接入 "
            "SurgicalLLMAgent。"
        )

    else:

        print("\n" + "=" * 60)
        print("[FAIL] DeepSeek 没有产生 Tool Call")
        print("=" * 60)

        print("\n模型直接回答内容：")

        try:
            print(response.output_text)
        except Exception:
            print(response.output)


except Exception as e:

    print("\n" + "=" * 60)
    print("[ERROR] DeepSeek API 调用失败")
    print("=" * 60)

    print(type(e).__name__)
    print(e)
