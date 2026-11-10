import os
import sys
from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY")
if "pytest" in sys.modules and os.getenv("RUN_DEEPSEEK_API_TESTS") != "1":
    import pytest
    pytest.skip("external paid DeepSeek test; set RUN_DEEPSEEK_API_TESTS=1", allow_module_level=True)
if not api_key:
    raise RuntimeError("未设置 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    timeout=30.0,
    max_retries=0,
)

# 若该模型名报错，请按 DeepSeek 当前官方文档或控制台修改。
model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

print("请求地址：", client.base_url)
print("请求模型：", model)

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "计算 17 加 25，只输出数字。",
        }
    ],
)

print("响应ID：", response.id)
print("响应模型：", response.model)
print("回答：", response.choices[0].message.content)
