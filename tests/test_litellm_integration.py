#!/usr/bin/env python3
"""
测试LiteLLM集成

这个脚本测试Shinka的LiteLLM集成是否正常工作
"""

import os
import sys
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from shinka.llm import LLMClient
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_multiple_models():
    """测试列表中的每一个模型"""
    print("\n" + "=" * 60)
    print("测试: 遍历测试所有指定模型")
    print("=" * 60)
    
    # 使用 LiteLLM 实际支持的 Moonshot 模型
    models_to_test = [
        # "gpt-5",
        "deepseek-chat",
        # "deepseek-reasoner", 
        # "claude-sonnet-4-5",
        # "gemini/gemini-2.5-pro",
        # "xai/grok-3",
        # "glm-4.5",
        # "volcengine/doubao-1-5-thinking-pro-250415",
        # "volcengine/doubao-seed-1-6-thinking-250715",
        # "moonshot/kimi-latest",
        # "moonshot/kimi-k2-thinking",
        # "dashscope/qwen3-coder-plus",
        # "dashscope/qwen3-max",
        # "dashscope/qwen-plus"

    ]
    
    results = {}
    
    for model in models_to_test:
        print(f"\n正在测试模型: {model} ...")
        try:
            # 为每个模型创建一个独立的客户端实例
            llm = LLMClient(
                model_names=model,
                temperatures=0.7,
                max_tokens=50,
                verbose=True
            )
            
            result = llm.query(
                msg="Hello, are you working?",
                system_msg="You are a helpful assistant."
            )
            
            if result:
                print(f"✓ 模型 {model} 测试成功!")
                print(f"  回复: {result.content}")
                results[model] = True
            else:
                print(f"✗ 模型 {model} 测试失败 (无返回结果)")
                results[model] = False
                
        except Exception as e:
            print(f"✗ 模型 {model} 测试异常: {str(e)}")
            results[model] = False

    # 详细总结
    print("\n" + "=" * 60)
    print("详细测试结果")
    print("=" * 60)
    for model, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status}: {model}")
        
    return all(results.values())

def main():
    """运行测试"""
    print("\n" + "=" * 60)
    print("开始测试Shinka LiteLLM集成")
    print("=" * 60)
    
    try:
        success = test_multiple_models()
    except Exception as e:
        print(f"\n✗ 测试执行异常: {str(e)}")
        import traceback
        traceback.print_exc()
        success = False
    
    # 最终总结
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有模型测试通过!")
    else:
        print("✗ 部分模型测试失败")

if __name__ == "__main__":
    main()
