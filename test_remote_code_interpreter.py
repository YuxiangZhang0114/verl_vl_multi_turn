import asyncio
import json
import time
from verl.tools.schemas import OpenAIFunctionToolSchema
from AIME.tools.remote_code_interpreter import RemoteCodeInterpreter

async def test_remote_code_interpreter():
    """测试远程代码解释器的基本功能"""
    
    # 创建工具配置
    config = {
        "service_url": "http://localhost:8001",
        "timeout": 30,
        "max_retries": 3,
        "retry_delay": 1.0,
        "session_timeout": 1800,
        "reuse_sessions": True
    }
    
    # 创建工具 schema
    tool_schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "code_interpreter",
            "description": "Execute Python code remotely",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    )
    
    # 创建远程代码解释器实例
    interpreter = RemoteCodeInterpreter(config, tool_schema)
    
    print("=== 测试远程代码解释器 ===")
    
    # 测试 1: 创建实例
    print("\n1. 测试创建实例...")
    try:
        instance_id = await interpreter.create()
        print(f"✅ 成功创建实例: {instance_id}")
    except Exception as e:
        print(f"❌ 创建实例失败: {e}")
        return
    
    # 测试 2: 执行简单的数学计算
    print("\n2. 测试执行简单计算...")
    try:
        code = "result = 2 + 3\nprint(f'2 + 3 = {result}')"
        result, _, _ = await interpreter.execute(instance_id, {"code": code})
        print(f"✅ 执行结果: {result}")
    except Exception as e:
        print(f"❌ 执行失败: {e}")
    
    # 测试 3: 执行带有图像输出的代码
    print("\n3. 测试执行图像生成代码...")
    try:
        code = """
import matplotlib.pyplot as plt
import numpy as np

# 生成简单的图表
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sin Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
"""
        result, _, _ = await interpreter.execute(instance_id, {"code": code})
        print(f"✅ 图像生成执行完成")
        print(f"结果类型: {type(result)}")
        if isinstance(result, list):
            print(f"结果包含 {len(result)} 个元素")
            for i, item in enumerate(result):
                print(f"  元素 {i}: {item.get('type', 'unknown')}")
    except Exception as e:
        print(f"❌ 图像生成执行失败: {e}")
    
    # 测试 4: 执行错误代码
    print("\n4. 测试执行错误代码...")
    try:
        code = "undefined_variable + 1"
        result, _, _ = await interpreter.execute(instance_id, {"code": code})
        print(f"✅ 错误处理正常: {result}")
    except Exception as e:
        print(f"❌ 错误处理失败: {e}")
    
    # 测试 5: 执行空代码
    print("\n5. 测试执行空代码...")
    try:
        result, _, _ = await interpreter.execute(instance_id, {"code": ""})
        print(f"✅ 空代码处理: {result}")
    except Exception as e:
        print(f"❌ 空代码处理失败: {e}")
    
    # 测试 6: 测试会话持久性
    print("\n6. 测试会话持久性...")
    try:
        # 定义一个变量
        code1 = "test_var = 'Hello, World!'"
        result1, _, _ = await interpreter.execute(instance_id, {"code": code1})
        
        # 在另一个代码块中使用该变量
        code2 = "print(test_var)"
        result2, _, _ = await interpreter.execute(instance_id, {"code": code2})
        print(f"✅ 会话持久性正常: {result2}")
    except Exception as e:
        print(f"❌ 会话持久性测试失败: {e}")
    
    # 测试 7: 释放实例
    print("\n7. 测试释放实例...")
    try:
        await interpreter.release(instance_id)
        print(f"✅ 成功释放实例: {instance_id}")
    except Exception as e:
        print(f"❌ 释放实例失败: {e}")
    
    print("\n=== 测试完成 ===")

def test_utility_functions():
    """测试工具函数"""
    print("\n=== 测试工具函数 ===")
    
    from AIME.tools.remote_code_interpreter import base64_to_image, scaling_image_by_max_size
    from PIL import Image
    import base64
    from io import BytesIO
    
    # 创建一个简单的测试图像
    print("\n1. 测试图像缩放函数...")
    try:
        # 创建一个 100x100 的红色图像
        test_image = Image.new('RGB', (100, 50), color='red')
        
        # 测试缩放
        scaled_image = scaling_image_by_max_size(test_image, max_size=50)
        print(f"✅ 原图大小: {test_image.size}, 缩放后大小: {scaled_image.size}")
        
        # 测试不需要缩放的情况
        small_image = Image.new('RGB', (30, 30), color='blue')
        unchanged_image = scaling_image_by_max_size(small_image, max_size=50)
        print(f"✅ 小图大小: {small_image.size}, 处理后大小: {unchanged_image.size}")
        
    except Exception as e:
        print(f"❌ 图像缩放测试失败: {e}")

async def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 配置错误的服务URL
    config = {
        "service_url": "http://localhost:9999",  # 错误的端口
        "timeout": 5,
        "max_retries": 2,
        "retry_delay": 0.5,
    }
    
    tool_schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "code_interpreter",
            "description": "Execute Python code remotely",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    )
    
    interpreter = RemoteCodeInterpreter(config, tool_schema)
    
    print("\n1. 测试连接错误处理...")
    try:
        instance_id = await interpreter.create()
        print(f"❌ 应该连接失败但却成功了: {instance_id}")
    except Exception as e:
        print(f"✅ 正确处理连接错误: {str(e)[:100]}...")

def main():
    """主函数"""
    print("开始测试远程代码解释器...")
    
    # 测试工具函数
    test_utility_functions()
    
    # 测试错误处理
    asyncio.run(test_error_handling())
    
    # 测试主要功能（需要远程服务运行）
    print("\n注意：以下测试需要远程服务在 http://localhost:8001 运行")
    user_input = input("是否继续测试远程功能？(y/n): ").strip().lower()
    
    if user_input == 'y':
        asyncio.run(test_remote_code_interpreter())
    else:
        print("跳过远程功能测试")
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    main()