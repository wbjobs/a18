@echo off
echo ========================================
echo  边缘计算联邦学习服务器
echo ========================================
echo.
echo [1/3] 检查配置...
python -c "import sys; print('Python版本:', sys.version)"
echo.
echo [2/3] 创建必要目录...
python -c "from config import Config; Config.ensure_dirs(); print('目录已创建')"
echo.
echo [3/3] 启动Flask服务器...
echo 服务器地址: http://localhost:5000
echo Dashboard: http://localhost:5000
echo.
echo 按 Ctrl+C 停止服务器
echo ========================================
echo.
cd server
python app.py
pause
