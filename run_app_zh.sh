#!/bin/bash

echo "正在启动 Picture Sherlock..."
echo "========================================"

# 从.env文件读取Python路径
if [ -f .env ]; then
    source .env
fi

# 检查Python路径是否已设置
if [ -z "$PYTHON_PATH" ] || [ ! -f "$PYTHON_PATH" ]; then
    echo "未找到有效的Python路径。请设置正确的Python解释器路径。"
    echo "您可以通过以下方式设置Python路径："
    echo "1. 编辑项目根目录下的.env文件，设置PYTHON_PATH变量"
    echo "2. 或者现在手动输入Python解释器的完整路径"
    echo ""
    read -p "请输入Python解释器的完整路径（回车跳过，使用系统默认Python）: " user_python_path
    
    if [ ! -z "$user_python_path" ]; then
        if [ -f "$user_python_path" ]; then
            # 更新.env文件
            if grep -q "^PYTHON_PATH=" .env; then
                sed -i "s|^PYTHON_PATH=.*|PYTHON_PATH=$user_python_path|" .env
            else
                echo "PYTHON_PATH=$user_python_path" >> .env
            fi
            PYTHON_PATH=$user_python_path
            echo "Python路径已更新至: $PYTHON_PATH"
        else
            echo "错误: 指定的路径 '$user_python_path' 不存在或不是一个有效的文件。"
            echo "将使用系统默认的Python。"
            PYTHON_PATH=$(which python3 2>/dev/null || which python)
        fi
    else
        # 使用系统默认Python
        PYTHON_PATH=$(which python3 2>/dev/null || which python)
        echo "将使用系统默认Python: $PYTHON_PATH"
    fi
fi

# 确保我们有一个有效的Python路径
if [ -z "$PYTHON_PATH" ] || [ ! -f "$PYTHON_PATH" ]; then
    if command -v python3 &>/dev/null; then
        PYTHON_PATH=$(which python3)
    elif command -v python &>/dev/null; then
        PYTHON_PATH=$(which python)
    else
        echo "错误: 未找到Python解释器。请安装Python 3.7+并重试。"
        exit 1
    fi
fi

echo "使用Python路径: $PYTHON_PATH"

# 检查Python版本
python_version=$("$PYTHON_PATH" -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
echo "Python版本: $python_version"

# 确认Python版本 >= 3.6
if [[ ! "$python_version" =~ ^3\.[6-9]|^3\.[1-9][0-9] ]]; then
    echo "警告: 推荐使用Python 3.6或更高版本。当前版本: $python_version"
    read -p "是否继续? (y/n): " continue_anyway
    if [[ ! "$continue_anyway" =~ ^[Yy] ]]; then
        echo "已取消。请安装Python 3.6+并重试。"
        exit 1
    fi
fi

# 检查是否有虚拟环境
venv_path=""
if [ -f "venv/bin/activate" ]; then
    echo "检测到虚拟环境。"
    venv_path="venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    echo "检测到虚拟环境。"
    venv_path=".venv/bin/activate"
fi

if [ ! -z "$venv_path" ]; then
    echo "激活虚拟环境..."
    source "$venv_path"
fi

# 检查依赖项
echo "检查依赖项..."
if ! "$PYTHON_PATH" -c "import streamlit" &>/dev/null; then
    echo "未安装所需的依赖项。正在安装..."
    
    # 检查requirements.txt文件是否存在
    if [ -f "requirements.txt" ]; then
        "$PYTHON_PATH" -m pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "安装依赖项失败。请查看错误信息并手动安装。"
            exit 1
        fi
    else
        echo "警告: 未找到requirements.txt文件。安装核心依赖..."
        "$PYTHON_PATH" -m pip install streamlit
        if [ $? -ne 0 ]; then
            echo "安装依赖项失败。请查看错误信息并手动安装。"
            exit 1
        fi
    fi
    
    echo "依赖项安装完成。"
else
    echo "所有必需的依赖项已安装。"
fi

# 确保脚本有执行权限
chmod +x "$0"

# 运行应用
echo "启动Picture Sherlock应用..."
"$PYTHON_PATH" -m streamlit run picture_sherlock/app.py -- --lang zh "$@" 