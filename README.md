# 边缘计算联邦学习框架与模型水印

## 项目架构

```
├── server/              # 中心服务器 (Flask)
│   ├── app.py           # Flask主应用
│   ├── federated.py     # 联邦学习核心 (FedAvg + 差分隐私)
│   ├── watermark.py     # 模型水印系统
│   ├── models.py        # 模型定义
│   ├── monitor.py       # 客户端监控
│   ├── static/          # Dashboard静态资源
│   └── templates/       # Dashboard模板
├── client/              # 边缘客户端 (Raspberry Pi模拟)
│   ├── client.py        # 客户端主程序
│   ├── dataset.py       # CIFAR-10变体数据集
│   ├── trainer.py       # 本地训练器
│   └── tflite_converter.py  # TensorFlow Lite转换
├── tests/               # 测试用例
└── scripts/             # 模拟脚本
    └── simulate_clients.py  # 多客户端模拟
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动服务器：
```bash
python server/app.py
```

3. 启动客户端（模拟多个边缘设备）：
```bash
python scripts/simulate_clients.py --num_clients 5
```

4. 访问Dashboard：
打开浏览器访问 http://localhost:5000

## 核心功能

- **联邦学习**：FedAvg聚合算法，客户端只上传梯度/权重更新
- **差分隐私**：ε=1.0的拉普拉斯噪声注入
- **模型水印**：后门触发式水印，用于模型盗版验证
- **实时监控**：客户端离线率、更新延迟、贡献度可视化
