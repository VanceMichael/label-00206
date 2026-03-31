# 平面桁架有限元分析平台 - 用户操作手册

## 1. 系统概述

本平台是一个基于 Python 的平面桁架结构有限元分析系统，支持：

- 任意几何形状的平面桁架结构建模
- 两种单元类型：桁架杆单元（TrussElement）和 CST 平面应力三角形单元（CSTElement）
- 静力分析（位移、应力、反力）
- 动力分析（地震响应、时程分析）
- 模态分析（固有频率、振型）
- 结果可视化（变形图、应力云图、动画）
- 稀疏矩阵和迭代求解器支持（适用于大规模问题）

## 2. 安装与运行

### 2.1 环境要求

- Python 3.11+
- 操作系统：Windows / macOS / Linux

### 2.2 安装步骤

```bash
# 克隆项目
git clone <repository_url>
cd fem-truss-platform

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2.3 运行方式

#### 命令行模式

```bash
# 运行简支桁架示例
python -m fem_truss.main --example simple_truss

# 运行刚架示例
python -m fem_truss.main --example frame_truss

# 指定输出目录
python -m fem_truss.main --example simple_truss --output results
```

#### Web 界面模式

```bash
python -m fem_truss.main --web
```

访问 http://localhost:8081 使用 Web 界面。

#### Docker 模式

```bash
docker-compose up --build -d
```

访问 http://localhost:8081

## 3. Jupyter Notebook 交互式使用

### 3.1 启用交互式绘图

在 Jupyter Notebook 中使用交互式绘图功能，推荐以下设置：

```python
# 方式一：使用 notebook 后端（推荐）
%matplotlib notebook

# 方式二：使用 widget 后端（需安装 ipympl）
# pip install ipympl
%matplotlib widget
```

### 3.2 交互式绘图示例

```python
%matplotlib notebook

from fem_truss.core import TrussStructure, Material
from fem_truss.solver import StaticSolver
from fem_truss.visualization import TrussPlotter
import matplotlib.pyplot as plt

# 创建结构
structure = TrussStructure()
steel = Material(id=1, name="Steel", E=2.06e11, rho=7850)
structure.add_material(steel)

# 添加节点和单元...
structure.add_node(1, 0.0, 0.0)
structure.add_node(2, 4.0, 0.0)
structure.add_node(3, 2.0, 2.0)
structure.add_element(1, 1, 2, material_id=1, area=0.001)
structure.add_element(2, 1, 3, material_id=1, area=0.001)
structure.add_element(3, 2, 3, material_id=1, area=0.001)
structure.apply_boundary(1, fix_x=True, fix_y=True)
structure.apply_boundary(2, fix_x=False, fix_y=True)
structure.apply_load(3, fx=0, fy=-10000)

# 求解
solver = StaticSolver(structure)
result = solver.solve()

# 交互式绘图（设置 save=False 以在 notebook 中显示）
plotter = TrussPlotter(structure, output_dir='output')
fig = plotter.plot_deformation(result.displacements, save=False)
plt.show()  # 显示交互式图形
```

### 3.3 交互式模式的优势

- 支持鼠标缩放和平移
- 可实时调整视图范围
- 适合探索性数据分析和结果检查
- 可在图形上添加标注

### 3.4 非交互式模式

对于批量处理或 Web 服务器环境，使用默认的非交互式模式：

```python
# 自动保存图片到文件
plotter.plot_structure()  # 默认 save=True
plotter.plot_deformation(result.displacements)
plotter.plot_stress_contour(result.stresses)
```

## 4. 数据文件格式

### 4.1 节点数据 (nodes.csv)

| 列名 | 类型 | 说明 |
|------|------|------|
| id | int | 节点编号 |
| x | float | X 坐标 (m) |
| y | float | Y 坐标 (m) |

示例：
```csv
id,x,y
1,0.0,0.0
2,4.0,0.0
3,2.0,2.0
```

### 4.2 单元数据 (elements.csv)

| 列名 | 类型 | 说明 |
|------|------|------|
| id | int | 单元编号 |
| node_i | int | 起始节点编号 |
| node_j | int | 终止节点编号 |
| material_id | int | 材料编号 |
| area | float | 截面面积 (m²) |

示例：
```csv
id,node_i,node_j,material_id,area
1,1,2,1,0.001
2,1,3,1,0.001
```

### 4.3 CST 单元数据 (cst_elements.csv)

| 列名 | 类型 | 说明 |
|------|------|------|
| id | int | 单元编号 |
| node_i | int | 节点 1 编号 |
| node_j | int | 节点 2 编号 |
| node_k | int | 节点 3 编号 |
| material_id | int | 材料编号 |
| thickness | float | 单元厚度 (m) |

示例：
```csv
id,node_i,node_j,node_k,material_id,thickness
1,1,2,3,1,0.01
```

### 4.4 材料数据 (materials.csv)

| 列名 | 类型 | 说明 |
|------|------|------|
| id | int | 材料编号 |
| name | str | 材料名称 |
| E | float | 弹性模量 (Pa) |
| nu | float | 泊松比（可选，默认 0.3）|
| rho | float | 密度 (kg/m³)（可选，默认 7850）|

示例：
```csv
id,name,E,nu,rho
1,Q235钢,2.06e11,0.3,7850
```

### 4.5 荷载数据 (loads.csv)

| 列名 | 类型 | 说明 |
|------|------|------|
| node_id | int | 作用节点编号 |
| fx | float | X 方向力 (N) |
| fy | float | Y 方向力 (N) |

示例：
```csv
node_id,fx,fy
3,0,-10000
```

### 4.6 边界条件 (boundaries.csv)

| 列名 | 类型 | 说明 |
|------|------|------|
| node_id | int | 节点编号 |
| fix_x | bool | 是否约束 X 方向 |
| fix_y | bool | 是否约束 Y 方向 |

示例：
```csv
node_id,fix_x,fix_y
1,True,True
2,False,True
```

### 4.7 地震波数据 (seismic.csv)

| 列名 | 类型 | 说明 |
|------|------|------|
| time | float | 时间 (s) |
| acceleration | float | 加速度 (m/s²) |

## 5. Web 界面使用

### 4.1 创建结构

1. 在左侧输入区域输入 JSON 格式的结构数据
2. 或点击"加载示例"按钮加载预设结构
3. 点击"创建结构"按钮

JSON 格式示例：
```json
{
  "materials": [
    {"id": 1, "name": "Steel", "E": 2.06e11, "rho": 7850}
  ],
  "nodes": [
    {"id": 1, "x": 0, "y": 0},
    {"id": 2, "x": 4, "y": 0},
    {"id": 3, "x": 2, "y": 2}
  ],
  "elements": [
    {"id": 1, "node_i": 1, "node_j": 2, "material_id": 1, "area": 0.001},
    {"id": 2, "node_i": 1, "node_j": 3, "material_id": 1, "area": 0.001},
    {"id": 3, "node_i": 2, "node_j": 3, "material_id": 1, "area": 0.001}
  ],
  "loads": [
    {"node_id": 3, "fx": 0, "fy": -10000}
  ],
  "boundaries": [
    {"node_id": 1, "fix_x": true, "fix_y": true},
    {"node_id": 2, "fix_x": false, "fix_y": true}
  ]
}
```

### 4.2 执行分析

- **静力分析**：点击"执行静力分析"按钮
- **动力分析**：切换到"动力分析"标签，设置参数后点击执行
- **模态分析**：切换到"模态分析"标签，设置模态数后点击执行

### 4.3 查看结果

分析完成后，右侧会显示：
- 数值结果（最大位移、最大应力等）
- 可视化图片（结构图、变形图、应力云图）
- 下载链接（CSV 结果文件、动画 GIF）

## 6. Python API 使用

### 6.1 创建结构

```python
from fem_truss.core import TrussStructure, Material

# 创建结构（小规模问题使用默认配置）
structure = TrussStructure()

# 大规模问题启用稀疏矩阵
# structure = TrussStructure(use_sparse=True)

# 添加材料
steel = Material(id=1, name="Steel", E=2.06e11, rho=7850)
structure.add_material(steel)

# 添加节点
structure.add_node(1, 0.0, 0.0)
structure.add_node(2, 4.0, 0.0)
structure.add_node(3, 2.0, 2.0)

# 添加桁架单元
structure.add_element(1, 1, 2, material_id=1, area=0.001)
structure.add_element(2, 1, 3, material_id=1, area=0.001)
structure.add_element(3, 2, 3, material_id=1, area=0.001)

# 或添加 CST 三角形单元
# structure.add_cst_element(1, 1, 2, 3, material_id=1, thickness=0.01)

# 施加边界条件
structure.apply_boundary(1, fix_x=True, fix_y=True)
structure.apply_boundary(2, fix_x=False, fix_y=True)

# 施加荷载
structure.apply_load(3, fx=0, fy=-10000)
```

### 6.2 静力分析

```python
from fem_truss.solver import StaticSolver

# 小规模问题
solver = StaticSolver(structure)
result = solver.solve()

# 大规模问题使用迭代求解器
# solver = StaticSolver(structure, use_iterative=True)
# solver.set_iterative_params(tol=1e-10, maxiter=1000)
# result = solver.solve()

print(f"最大位移: {max(abs(result.displacements))} m")
print(f"最大应力: {max(abs(s) for s in result.stresses.values())} Pa")
```

### 6.3 动力分析

```python
import numpy as np
from fem_truss.solver import DynamicSolver

# 创建求解器
solver = DynamicSolver(structure)
solver.set_damping_ratio(0.05)

# 生成地震波
time = np.arange(0, 2, 0.01)
acceleration = 0.1 * 9.81 * np.sin(2 * np.pi * 2 * time)

# 求解
result = solver.solve(acceleration, time, direction='x')
```

### 6.4 可视化

```python
from fem_truss.visualization import TrussPlotter

plotter = TrussPlotter(structure, output_dir='output')

# 绘制结构图
plotter.plot_structure()

# 绘制变形图
plotter.plot_deformation(result.displacements)

# 绘制应力云图
plotter.plot_stress_contour(result.stresses)

# 创建动画
plotter.create_animation(dynamic_result)
```

## 7. 性能与规模

### 7.1 性能基准

以下是不同规模问题的典型求解耗时（参考值，实际性能因硬件而异）：

| 节点数 | 自由度 | 密集矩阵(s) | 稀疏矩阵(s) | 迭代求解(s) |
|--------|--------|-------------|-------------|-------------|
| 100    | 200    | 0.02        | 0.01        | 0.01        |
| 500    | 1000   | 0.5         | 0.05        | 0.03        |
| 1000   | 2000   | 4.0         | 0.15        | 0.08        |
| 2000   | 4000   | 35.0        | 0.6         | 0.3         |
| 5000   | 10000  | OOM         | 3.5         | 1.5         |

### 7.2 配置建议

- 节点数 < 500：使用默认配置（密集矩阵）
- 节点数 500-2000：启用稀疏矩阵 + 直接求解器
- 节点数 > 2000：启用稀疏矩阵 + 迭代求解器

```python
# 大规模问题配置示例
structure = TrussStructure(use_sparse=True)
# ... 添加节点和单元 ...

solver = StaticSolver(structure, use_iterative=True)
solver.set_iterative_params(tol=1e-10, maxiter=2000)
result = solver.solve()
```

### 7.3 运行基准测试

```bash
# 运行 1000 节点基准测试
python -m fem_truss.benchmark

# 运行完整基准测试（100-2000 节点）
python -m fem_truss.benchmark --full
```

## 8. 输出文件说明

### 8.1 displacements.csv

节点位移结果：

| 列名 | 说明 |
|------|------|
| node_id | 节点编号 |
| ux (m) | X 方向位移 |
| uy (m) | Y 方向位移 |
| magnitude (m) | 位移幅值 |

### 8.2 stresses.csv

单元应力结果：

| 列名 | 说明 |
|------|------|
| element_id | 单元编号 |
| stress (Pa) | 应力值 |
| stress (MPa) | 应力值（MPa）|
| strain | 应变值 |

### 8.3 可视化文件

- `structure.png` - 原始结构图
- `deformation.png` - 变形放大图
- `stress_contour.png` - 应力云图
- `animation.gif` - 位移时程动画

## 9. 常见问题

### Q1: 刚度矩阵奇异怎么办？

检查边界条件是否足够。平面桁架至少需要约束 3 个自由度以防止刚体运动。

### Q2: 计算结果不收敛？

- 检查单元连接是否正确
- 检查材料参数是否合理
- 对于动力分析，尝试减小时间步长

### Q3: 如何处理大规模模型？

启用稀疏矩阵和迭代求解器：

```python
structure = TrussStructure(use_sparse=True)
solver = StaticSolver(structure, use_iterative=True)
```

### Q4: CST 单元和桁架单元有什么区别？

- 桁架单元（TrussElement）：一维杆单元，仅承受轴向力，适用于桁架结构
- CST 单元（CSTElement）：二维三角形单元，承受平面应力，适用于薄板、膜结构

### Q5: 如何在 Notebook 中使用交互式绘图？

在代码开头添加：
```python
%matplotlib notebook
```
然后设置 `save=False` 以在 notebook 中显示图形。

## 10. 技术支持

如有问题，请提交 Issue 或联系开发团队。
