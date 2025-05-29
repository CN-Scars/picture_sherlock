"""
启动脚本：直接运行此脚本启动Streamlit应用
"""
import os
import subprocess
import sys
from pathlib import Path

def main():
    """运行Streamlit应用"""
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent.absolute()
    app_path = os.path.join(current_dir, "app.py")
    
    # 将父目录添加到Python路径中，以便能够正确导入模块
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.append(str(parent_dir))
    
    # 检查app.py是否存在
    if not os.path.isfile(app_path):
        print(f"错误: 未找到应用文件 '{app_path}'")
        sys.exit(1)
    
    print("启动Picture Sherlock...")
    print(f"应用路径: {app_path}")
    
    # 启动streamlit应用
    try:
        print("运行环境:", sys.executable)
        print("命令:", [sys.executable, "-m", "streamlit", "run", str(app_path)])
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    except KeyboardInterrupt:
        print("\n应用已停止")
    except Exception as e:
        print(f"启动应用时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 