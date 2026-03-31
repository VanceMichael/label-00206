# 平面桁架结构有限元分析平台

## How to Run

### 快速开始

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行简支桁架验证算例
python -m fem_truss.main --example simple_truss --output output

# 4. 运行刚架结构验证算例
python -m fem_truss.main --example frame_truss --output output

# 5. 启动 Web 界面
python -m fem_truss.main --web

# 6. 运行单元测试
pytest tests/ -v

# 7. Docker 方式运行
docker-compose up --build -d
```

### 验证算例运行说明

**算例1：简支桁架**
```bash
python -m fem_truss.main --example simple_truss
```
预期输出：
- 最大位移: 0.2378 mm
- 最大应力: 7.07 MPa
- 输出文件: output/displacements.csv, output/stresses.csv, output/animation.gif

**算例2：刚架结构**
```bash
python -m fem_truss.main --example frame_truss
```
预期输出：
- 最大位移: 0.892 mm
- 最大应力: 15.8 MPa

### Jupyter Notebook 使用

```python
from fem_truss.core import TrussStructure, Material
from fem_truss.visualization import TrussPlotter

# 创建结构
structure = TrussStructure()
structure.add_material(Material(id=1, name="Steel", E=2.06e11))
structure.add_node(1, 0, 0)
structure.add_node(2, 4, 0)
structure.add_node(3, 2, 2)
# ... 添加单元、荷载、边界条件

# 直接调用结构的求解方法
result = structure.solve_static()

# 可视化（在 Notebook 中自动显示）
%matplotlib inline
plotter = TrussPlotter(structure)
plotter.plot_deformation(result.displacements)
```

## Services

| 服务 | 端口 | 说明 |
|------|------|------|
| fem-truss | 8081 | 有限元分析平台 Web 界面 |

访问地址: http://localhost:8081

## 测试账号

本项目为科学计算平台，无需登录账号。

## 题目内容

开发一个功能完整、通用的平面桁架结构有限元分析平台，能够支持任意几何形状、材料属性和荷载条件的平面桁架结构分析，实现从结构建模、静力分析到动力响应分析的全流程自动化处理，并生成规范的分析结果报告与可视化展示。

---

## 项目简介

本平台是一个基于 Python 的平面桁架结构有限元分析系统，采用面向对象设计，实现了从结构建模、静力分析到动力响应分析的全流程自动化处理。

### 核心功能

- **结构建模**：支持任意几何形状的平面桁架结构定义
- **静力分析**：基于有限元方法求解结构位移和应力
- **动力分析**：基于 Newmark-β 法求解地震动响应
- **可视化**：结构变形图、应力云图、位移时程动画
- **结果输出**：CSV 格式的位移和应力结果文件

### 技术栈

- Python 3.11+
- NumPy - 核心数值计算
- Pandas - 数据处理
- SciPy - 科学计算
- Matplotlib - 可视化
- Flask - Web 界面
- Pytest - 单元测试

### 项目结构

```
fem-truss-platform/
├── fem_truss/                 # 核心包
│   ├── __init__.py
│   ├── main.py               # 主程序入口
│   ├── core/                 # 核心类
│   │   ├── __init__.py
│   │   ├── element.py        # TrussElement 类
│   │   ├── structure.py      # TrussStructure 类
│   │   └── material.py       # 材料类
│   ├── io/                   # 数据输入输出
│   │   ├── __init__.py
│   │   ├── reader.py         # 数据读取
│   │   └── writer.py         # 结果输出
│   ├── solver/               # 求解器
│   │   ├── __init__.py
│   │   ├── static.py         # 静力求解
│   │   └── dynamic.py        # 动力求解
│   ├── visualization/        # 可视化
│   │   ├── __init__.py
│   │   └── plotter.py        # 绑图模块
│   ├── web/                  # Web 界面
│   │   ├── __init__.py
│   │   ├── app.py            # Flask 应用
│   │   ├── templates/        # HTML 模板
│   │   └── static/           # 静态资源
│   └── utils/                # 工具模块
│       ├── __init__.py
│       ├── logger.py         # 日志
│       └── exceptions.py     # 异常处理
├── data/                     # 示例数据
│   ├── simple_truss/         # 简支桁架算例
│   └── frame_truss/          # 刚架算例
├── tests/                    # 测试用例
├── output/                   # 输出目录
├── docs/                     # 文档
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### 验证算例

1. **简支桁架**：3 节点简支桁架，验证静力分析精度
2. **刚架结构**：多节点刚架，验证动力分析功能

### 许可证

MIT License
# label-00206
