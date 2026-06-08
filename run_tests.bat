@echo off
echo ========================================
echo  运行单元测试
echo ========================================
echo.

echo [1/3] 联邦学习算法测试...
python -m unittest tests.test_federated -v
echo.

echo [2/3] 模型水印测试...
python -m unittest tests.test_watermark -v
echo.

echo [3/3] 客户端监控测试...
python -m unittest tests.test_monitor -v
echo.

echo ========================================
echo  非服务器端测试完成
echo ========================================
echo.
echo 要运行完整的端到端测试，请先启动服务器，然后运行:
echo   python scripts\test_end_to_end.py
echo.
pause
