"""
运行门控跨模态注意力测试脚本
"""
import os
import sys
import subprocess
import time
import platform

def run_test():
    """运行测试脚本并显示结果"""
    print("=" * 60)
    print("开始测试门控跨模态注意力机制")
    print("=" * 60)
    
    # 检测操作系统
    system = platform.system()
    print(f"当前操作系统: {system}")
    
    # 运行测试脚本
    start_time = time.time()
    
    # 根据操作系统设置环境变量
    env = os.environ.copy()
    if system == "Windows":
        # 在Windows上设置字体相关环境变量
        env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        result = subprocess.run(["python", "test_gated_fusion.py"], 
                               capture_output=True, 
                               text=True,
                               encoding="utf-8",
                               env=env)
        end_time = time.time()
    except Exception as e:
        print(f"运行测试脚本时出错: {e}")
        return
    
    # 显示输出
    print("\n测试输出:")
    print("-" * 60)
    try:
        print(result.stdout)
    except UnicodeEncodeError:
        print("无法显示某些输出，可能包含不支持的字符")
    
    if result.stderr:
        print("\n错误信息:")
        print("-" * 60)
        try:
            print(result.stderr)
        except UnicodeEncodeError:
            print("无法显示某些错误信息，可能包含不支持的字符")
    
    # 检查是否生成了图像文件
    if os.path.exists("gate_coefficients.png"):
        print("\n成功生成门控系数分布图: gate_coefficients.png")
        # 检查文件大小，确保图像有效
        file_size = os.path.getsize("gate_coefficients.png")
        print(f"图像文件大小: {file_size} 字节")
        if file_size < 1000:
            print("警告: 图像文件过小，可能生成有误")
    else:
        print("\n警告: 未能生成门控系数分布图")
    
    print("\n测试完成，耗时: {:.2f}秒".format(end_time - start_time))
    print("=" * 60)
    
    # 提示用户运行比较脚本
    print("\n要运行完整的模型比较，请执行:")
    print("python compare_models.py")
    print("\n这将比较标准模型和门控模型在不同噪声级别下的性能")
    print("=" * 60)
    
    # 提示字体问题
    print("\n注意: 如果图像中的中文显示为方框，这是由于缺少中文字体支持")
    print("解决方案:")
    if system == "Windows":
        print("1. 确保系统安装了中文字体（如宋体、黑体等）")
        print("2. 重新运行测试脚本")
    elif system == "Linux":
        print("1. 安装中文字体: sudo apt-get install fonts-wqy-microhei")
        print("2. 重新运行测试脚本")
    elif system == "Darwin":  # macOS
        print("1. 确保系统安装了中文字体")
        print("2. 重新运行测试脚本")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
