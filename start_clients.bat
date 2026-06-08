@echo off
echo ========================================
echo  启动多个边缘客户端模拟
echo ========================================
echo.
set /p NUM_CLIENTS=请输入客户端数量 (默认5): 
if "%NUM_CLIENTS%"=="" set NUM_CLIENTS=5

set /p ROUNDS=请输入训练回合数 (默认10): 
if "%ROUNDS%"=="" set ROUNDS=10

set /p DELAY=请输入回合间隔秒数 (默认15): 
if "%DELAY%"=="" set DELAY=15

echo.
echo 将启动 %NUM_CLIENTS% 个客户端，每客户端 %ROUNDS% 回合
echo.
echo 确保服务器已在 http://localhost:5000 运行
echo 按 Ctrl+C 停止所有客户端
echo ========================================
echo.

cd scripts
python simulate_clients.py --num_clients %NUM_CLIENTS% --rounds %ROUNDS% --delay %DELAY%

pause
