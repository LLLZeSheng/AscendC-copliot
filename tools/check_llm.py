# 首先安装依赖：pip3 install openai
import os
from openai import OpenAI

def deepseek_chat_demo():
    """调用 DeepSeek API 进行简单对话的示例函数"""
    # 直接使用你提供的 API Key（仅用于测试）
    api_key = "sk-d1e612e48d654731bffff6cbf35fdfd8"
    if not api_key:
        raise ValueError("API Key 不能为空")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"  # 必须带 /v1，否则调用失败
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            stream=False
        )

        # 输出回复内容
        reply_content = response.choices[0].message.content
        print("DeepSeek 回复：", reply_content)
        return reply_content
    
    except Exception as e:
        print(f"调用 API 出错：{e}")
        return None

# 执行函数
if __name__ == "__main__":
    deepseek_chat_demo()