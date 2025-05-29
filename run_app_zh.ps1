# 设置编码为UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "正在启动 Picture Sherlock..."
Write-Host "========================================"

# 从.env文件读取Python路径
$PYTHON_PATH = $null
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^PYTHON_PATH=(.*)$") {
            $PYTHON_PATH = $matches[1].Trim('"')
            # 处理环境变量
            $PYTHON_PATH = $PYTHON_PATH -replace '%USERPROFILE%', $env:USERPROFILE
            $PYTHON_PATH = $PYTHON_PATH -replace '\$HOME', $HOME
        }
    }
}

# 检查Python路径是否已设置
if ([string]::IsNullOrEmpty($PYTHON_PATH) -or -not (Test-Path $PYTHON_PATH)) {
    Write-Host "未找到有效的Python路径。请设置正确的Python解释器路径。"
    Write-Host "您可以通过以下方式设置Python路径："
    Write-Host "1. 编辑项目根目录下的.env文件，设置PYTHON_PATH变量"
    Write-Host "2. 或者现在手动输入Python解释器的完整路径"
    Write-Host ""
    $user_python_path = Read-Host "请输入Python解释器的完整路径（回车跳过，使用系统默认Python）"
    
    if (-not [string]::IsNullOrEmpty($user_python_path)) {
        if (Test-Path $user_python_path) {
            # 更新.env文件
            $env_content = Get-Content ".env"
            $python_path_updated = $false
            
            for ($i = 0; $i -lt $env_content.Count; $i++) {
                if ($env_content[$i] -match "^PYTHON_PATH=") {
                    $env_content[$i] = "PYTHON_PATH=`"$user_python_path`""
                    $python_path_updated = $true
                    break
                }
            }
            
            if (-not $python_path_updated) {
                $env_content += "PYTHON_PATH=`"$user_python_path`""
            }
            
            $env_content | Set-Content ".env"
            $PYTHON_PATH = $user_python_path
            Write-Host "Python路径已更新至: $PYTHON_PATH"
        } else {
            Write-Host "错误: 指定的路径 '$user_python_path' 不存在或不是一个有效的文件。"
            Write-Host "将使用系统默认的Python。"
            $PYTHON_PATH = $null
        }
    } else {
        # 使用系统默认Python
        $PYTHON_PATH = $null
    }
}

# 确保我们有一个有效的Python路径
if ([string]::IsNullOrEmpty($PYTHON_PATH) -or -not (Test-Path $PYTHON_PATH)) {
    # 查找Python可执行文件
    $python_cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $python_cmd) {
        $python_cmd = Get-Command python3 -ErrorAction SilentlyContinue
    }
    if ($null -eq $python_cmd) {
        $python_cmd = Get-Command py -ErrorAction SilentlyContinue
    }
    
    if ($null -ne $python_cmd) {
        $PYTHON_PATH = $python_cmd.Path
        Write-Host "将使用系统默认Python: $PYTHON_PATH"
    } else {
        Write-Host "错误: 未找到Python解释器。请安装Python 3.7+并重试。"
        exit 1
    }
}

Write-Host "使用Python路径: $PYTHON_PATH"

# 检查Python版本
$python_version = & $PYTHON_PATH -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))"
Write-Host "Python版本: $python_version"

# 确认Python版本 >= 3.6
if ($python_version -notmatch "^3\.[6-9]|^3\.[1-9][0-9]") {
    Write-Host "警告: 推荐使用Python 3.6或更高版本。当前版本: $python_version"
    $continue_anyway = Read-Host "是否继续? (y/n)"
    if ($continue_anyway -notmatch "^[Yy]") {
        Write-Host "已取消。请安装Python 3.6+并重试。"
        exit 1
    }
}

# 检查是否有虚拟环境
$venv_path = $null
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "检测到虚拟环境。"
    $venv_path = "venv\Scripts\Activate.ps1"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "检测到虚拟环境。"
    $venv_path = ".venv\Scripts\Activate.ps1"
}

if (-not [string]::IsNullOrEmpty($venv_path)) {
    Write-Host "激活虚拟环境..."
    . $venv_path
}

# 检查依赖项
Write-Host "检查依赖项..."
$has_streamlit = $false
try {
    & $PYTHON_PATH -c "import streamlit" 2>$null
    $has_streamlit = $LASTEXITCODE -eq 0
} catch {
    $has_streamlit = $false
}

if (-not $has_streamlit) {
    Write-Host "未安装所需的依赖项。正在安装..."
    
    # 检查requirements.txt文件是否存在
    if (Test-Path "requirements.txt") {
        & $PYTHON_PATH -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Host "安装依赖项失败。请查看错误信息并手动安装。"
            exit 1
        }
    } else {
        Write-Host "警告: 未找到requirements.txt文件。安装核心依赖..."
        & $PYTHON_PATH -m pip install streamlit
        if ($LASTEXITCODE -ne 0) {
            Write-Host "安装依赖项失败。请查看错误信息并手动安装。"
            exit 1
        }
    }
    
    Write-Host "依赖项安装完成。"
} else {
    Write-Host "所有必需的依赖项已安装。"
}

# 运行应用
Write-Host "启动Picture Sherlock应用..."
& $PYTHON_PATH -m streamlit run picture_sherlock/app.py -- --lang zh $args 