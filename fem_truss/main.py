"""
平面桁架有限元分析平台 - 主程序入口

使用方法:
    python -m fem_truss.main [--web] [--example NAME]
    
参数:
    --web       启动 Web 界面
    --example   运行示例分析 (simple_truss 或 frame_truss)
"""
import argparse
import numpy as np
from pathlib import Path

from .core import TrussStructure, Material
from .solver import StaticSolver, DynamicSolver
from .visualization import TrussPlotter
from .io import ResultWriter
from .utils import get_logger

logger = get_logger(__name__)


def create_simple_truss() -> TrussStructure:
    """创建简支桁架示例"""
    structure = TrussStructure()
    
    # 添加材料
    steel = Material(id=1, name="Q235钢", E=2.06e11, nu=0.3, rho=7850)
    structure.add_material(steel)
    
    # 添加节点 (三角形桁架)
    structure.add_node(1, 0.0, 0.0)
    structure.add_node(2, 4.0, 0.0)
    structure.add_node(3, 2.0, 2.0)
    
    # 添加单元
    structure.add_element(1, 1, 2, material_id=1, area=0.001)  # 底边
    structure.add_element(2, 1, 3, material_id=1, area=0.001)  # 左斜边
    structure.add_element(3, 2, 3, material_id=1, area=0.001)  # 右斜边
    
    # 施加边界条件
    structure.apply_boundary(1, fix_x=True, fix_y=True)   # 固定铰支座
    structure.apply_boundary(2, fix_x=False, fix_y=True)  # 滑动铰支座
    
    # 施加荷载
    structure.apply_load(3, fx=0, fy=-10000)  # 顶点向下 10kN
    
    return structure


def create_frame_truss() -> TrussStructure:
    """创建刚架桁架示例"""
    structure = TrussStructure()
    
    # 添加材料
    steel = Material(id=1, name="Q345钢", E=2.06e11, nu=0.3, rho=7850)
    structure.add_material(steel)
    
    # 添加节点 (2x2 网格)
    structure.add_node(1, 0.0, 0.0)
    structure.add_node(2, 3.0, 0.0)
    structure.add_node(3, 6.0, 0.0)
    structure.add_node(4, 0.0, 3.0)
    structure.add_node(5, 3.0, 3.0)
    structure.add_node(6, 6.0, 3.0)
    
    # 添加单元
    # 底边
    structure.add_element(1, 1, 2, material_id=1, area=0.002)
    structure.add_element(2, 2, 3, material_id=1, area=0.002)
    # 顶边
    structure.add_element(3, 4, 5, material_id=1, area=0.002)
    structure.add_element(4, 5, 6, material_id=1, area=0.002)
    # 竖杆
    structure.add_element(5, 1, 4, material_id=1, area=0.002)
    structure.add_element(6, 2, 5, material_id=1, area=0.002)
    structure.add_element(7, 3, 6, material_id=1, area=0.002)
    # 斜杆
    structure.add_element(8, 1, 5, material_id=1, area=0.0015)
    structure.add_element(9, 2, 4, material_id=1, area=0.0015)
    structure.add_element(10, 2, 6, material_id=1, area=0.0015)
    structure.add_element(11, 3, 5, material_id=1, area=0.0015)
    
    # 施加边界条件
    structure.apply_boundary(1, fix_x=True, fix_y=True)
    structure.apply_boundary(3, fix_x=False, fix_y=True)
    
    # 施加荷载
    structure.apply_load(4, fx=5000, fy=0)
    structure.apply_load(5, fx=0, fy=-20000)
    structure.apply_load(6, fx=5000, fy=0)
    
    return structure


def run_static_analysis(structure: TrussStructure, output_dir: str = 'output'):
    """运行静力分析"""
    logger.info("=" * 60)
    logger.info("开始静力分析")
    logger.info("=" * 60)
    
    print(structure.summary())
    
    # 求解
    solver = StaticSolver(structure)
    result = solver.solve()
    
    # 输出结果
    writer = ResultWriter(output_dir)
    node_ids = sorted(structure.nodes.keys())
    writer.write_displacements(result.displacements, node_ids)
    writer.write_stresses(result.stresses, result.strains)
    writer.write_reactions(result.reactions, structure.get_constrained_dofs(), node_ids)
    
    # 可视化
    plotter = TrussPlotter(structure, output_dir)
    plotter.plot_structure()
    plotter.plot_deformation(result.displacements)
    plotter.plot_stress_contour(result.stresses)
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("静力分析结果摘要")
    print("=" * 60)
    print(f"最大位移: {np.max(np.abs(result.displacements)) * 1000:.4f} mm")
    print(f"最大应力: {max(abs(s) for s in result.stresses.values()) / 1e6:.2f} MPa")
    print(f"结果文件已保存到: {output_dir}/")
    
    return result


def run_dynamic_analysis(structure: TrussStructure, output_dir: str = 'output',
                        seismic_file: str = None):
    """运行动力分析
    
    Args:
        structure: 桁架结构
        output_dir: 输出目录
        seismic_file: 地震波文件路径，默认使用 data/seismic/el_centro.csv
    """
    from .io import DataReader
    
    logger.info("=" * 60)
    logger.info("开始动力分析")
    logger.info("=" * 60)
    
    # 确定地震波文件路径
    if seismic_file is None:
        # 默认使用 El Centro 地震波
        base_dir = Path(__file__).parent.parent
        seismic_file = base_dir / 'data' / 'seismic' / 'el_centro.csv'
    
    seismic_path = Path(seismic_file)
    
    if seismic_path.exists():
        # 读取真实地震波数据
        logger.info(f"读取地震波文件: {seismic_path}")
        seismic_df = DataReader.read_seismic(str(seismic_path))
        time = seismic_df['time'].values
        acceleration = seismic_df['acceleration'].values * 9.81  # 转换为 m/s²
        logger.info(f"地震波时长: {time[-1]:.2f}s, 数据点数: {len(time)}")
    else:
        # 如果文件不存在，生成简谐地震波作为备选
        logger.warning(f"地震波文件不存在: {seismic_path}，使用生成的简谐波")
        duration = 2.0
        dt = 0.01
        time = np.arange(0, duration, dt)
        
        # 简谐波 + 衰减
        frequency = 2.0  # Hz
        amplitude = 0.1 * 9.81  # 0.1g
        acceleration = amplitude * np.sin(2 * np.pi * frequency * time)
        acceleration *= np.exp(-0.5 * time)  # 衰减
    
    # 求解
    solver = DynamicSolver(structure)
    solver.set_damping_ratio(0.05)
    result = solver.solve(acceleration, time, direction='x')
    
    # 输出结果
    writer = ResultWriter(output_dir)
    node_indices = list(range(min(3, structure.num_nodes)))
    writer.write_time_history(result, node_indices)
    
    # 可视化
    plotter = TrussPlotter(structure, output_dir)
    plotter.create_animation(result)
    plotter.plot_time_history(result, node_indices, direction='x')
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("动力分析结果摘要")
    print("=" * 60)
    print(f"分析时长: {time[-1]:.2f} s")
    print(f"时间步数: {len(time)}")
    print(f"最大位移: {np.max(np.abs(result.displacement_history)) * 1000:.4f} mm")
    print(f"动画已保存到: {output_dir}/animation.gif")
    
    return result


def run_modal_analysis(structure: TrussStructure):
    """运行模态分析"""
    logger.info("=" * 60)
    logger.info("开始模态分析")
    logger.info("=" * 60)
    
    solver = StaticSolver(structure)
    frequencies, mode_shapes = solver.modal_analysis(num_modes=6)
    
    print("\n" + "=" * 60)
    print("模态分析结果")
    print("=" * 60)
    for i, freq in enumerate(frequencies):
        period = 1 / freq if freq > 0 else float('inf')
        print(f"第 {i+1} 阶: 频率 = {freq:.4f} Hz, 周期 = {period:.4f} s")
    
    return frequencies, mode_shapes


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='平面桁架有限元分析平台')
    parser.add_argument('--web', action='store_true', help='启动 Web 界面')
    parser.add_argument('--example', type=str, choices=['simple_truss', 'frame_truss'],
                       default='simple_truss', help='运行示例分析')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    
    args = parser.parse_args()
    
    if args.web:
        # 启动 Web 服务
        from .web.app import app
        logger.info("启动 Web 服务...")
        app.run(host='0.0.0.0', port=8081, debug=False)
    else:
        # 运行示例分析
        logger.info(f"运行示例: {args.example}")
        
        # 创建输出目录
        Path(args.output).mkdir(exist_ok=True)
        
        # 创建结构
        if args.example == 'simple_truss':
            structure = create_simple_truss()
        else:
            structure = create_frame_truss()
        
        # 运行分析
        run_static_analysis(structure, args.output)
        run_modal_analysis(structure)
        run_dynamic_analysis(structure, args.output)
        
        print("\n" + "=" * 60)
        print("所有分析完成!")
        print("=" * 60)


if __name__ == '__main__':
    main()
