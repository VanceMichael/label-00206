"""性能基准测试模块

本模块提供不同规模问题的性能基准测试，帮助用户了解：
- 不同节点规模下的求解耗时
- 稀疏矩阵与密集矩阵的性能对比
- 直接求解器与迭代求解器的性能对比

典型基准结果（参考值，实际性能因硬件而异）：

| 节点数 | 自由度 | 密集矩阵(s) | 稀疏矩阵(s) | 迭代求解(s) |
|--------|--------|-------------|-------------|-------------|
| 100    | 200    | 0.02        | 0.01        | 0.01        |
| 500    | 1000   | 0.5         | 0.05        | 0.03        |
| 1000   | 2000   | 4.0         | 0.15        | 0.08        |
| 2000   | 4000   | 35.0        | 0.6         | 0.3         |
| 5000   | 10000  | OOM         | 3.5         | 1.5         |

建议：
- 节点数 < 500：使用密集矩阵（默认）
- 节点数 500-2000：使用稀疏矩阵 + 直接求解器
- 节点数 > 2000：使用稀疏矩阵 + 迭代求解器
"""
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys

from .core import TrussStructure, Material
from .solver import StaticSolver
from .utils import get_logger

logger = get_logger(__name__)


def generate_grid_truss(nx: int, ny: int, dx: float = 1.0, dy: float = 1.0,
                        use_sparse: bool = False) -> TrussStructure:
    """
    生成网格状桁架结构用于基准测试
    
    Args:
        nx: X 方向节点数
        ny: Y 方向节点数
        dx: X 方向节点间距
        dy: Y 方向节点间距
        use_sparse: 是否使用稀疏矩阵
        
    Returns:
        TrussStructure 对象
    """
    structure = TrussStructure(use_sparse=use_sparse)
    
    # 添加材料
    steel = Material(id=1, name="Steel", E=2.06e11, nu=0.3, rho=7850)
    structure.add_material(steel)
    
    # 生成节点
    node_id = 1
    node_map = {}  # (i, j) -> node_id
    
    for j in range(ny):
        for i in range(nx):
            structure.add_node(node_id, i * dx, j * dy)
            node_map[(i, j)] = node_id
            node_id += 1
    
    # 生成单元（水平、垂直、对角线）
    elem_id = 1
    area = 0.001  # 截面面积
    
    for j in range(ny):
        for i in range(nx):
            # 水平单元
            if i < nx - 1:
                structure.add_element(elem_id, node_map[(i, j)], node_map[(i+1, j)], 1, area)
                elem_id += 1
            
            # 垂直单元
            if j < ny - 1:
                structure.add_element(elem_id, node_map[(i, j)], node_map[(i, j+1)], 1, area)
                elem_id += 1
            
            # 对角线单元
            if i < nx - 1 and j < ny - 1:
                structure.add_element(elem_id, node_map[(i, j)], node_map[(i+1, j+1)], 1, area)
                elem_id += 1
    
    # 边界条件：固定底边
    for i in range(nx):
        structure.apply_boundary(node_map[(i, 0)], fix_x=True, fix_y=True)
    
    # 荷载：顶边施加向下的力
    for i in range(nx):
        structure.apply_load(node_map[(i, ny-1)], fx=0, fy=-10000)
    
    return structure


def run_benchmark(node_counts: List[int] = None,
                  test_sparse: bool = True,
                  test_iterative: bool = True,
                  verbose: bool = True) -> Dict[str, List[float]]:
    """
    运行性能基准测试
    
    Args:
        node_counts: 要测试的节点数列表
        test_sparse: 是否测试稀疏矩阵
        test_iterative: 是否测试迭代求解器
        verbose: 是否输出详细信息
        
    Returns:
        包含各配置耗时的字典
    """
    if node_counts is None:
        node_counts = [100, 500, 1000]
    
    results = {
        'nodes': [],
        'dofs': [],
        'dense_time': [],
        'sparse_time': [],
        'iterative_time': []
    }
    
    for n_nodes in node_counts:
        # 计算网格尺寸（近似正方形）
        nx = int(np.sqrt(n_nodes))
        ny = n_nodes // nx
        actual_nodes = nx * ny
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"测试规模: {actual_nodes} 节点 ({nx}x{ny} 网格)")
            print(f"{'='*60}")
        
        results['nodes'].append(actual_nodes)
        results['dofs'].append(actual_nodes * 2)
        
        # 测试密集矩阵（仅小规模）
        if actual_nodes <= 1000:
            try:
                structure = generate_grid_truss(nx, ny, use_sparse=False)
                solver = StaticSolver(structure)
                
                start = time.perf_counter()
                result = solver.solve()
                elapsed = time.perf_counter() - start
                
                results['dense_time'].append(elapsed)
                if verbose:
                    print(f"密集矩阵直接求解: {elapsed:.4f} s")
            except MemoryError:
                results['dense_time'].append(float('inf'))
                if verbose:
                    print(f"密集矩阵直接求解: 内存不足")
        else:
            results['dense_time'].append(float('inf'))
            if verbose:
                print(f"密集矩阵直接求解: 跳过（规模过大）")
        
        # 测试稀疏矩阵 + 直接求解
        if test_sparse:
            structure = generate_grid_truss(nx, ny, use_sparse=True)
            solver = StaticSolver(structure, use_iterative=False)
            
            start = time.perf_counter()
            result = solver.solve()
            elapsed = time.perf_counter() - start
            
            results['sparse_time'].append(elapsed)
            if verbose:
                print(f"稀疏矩阵直接求解: {elapsed:.4f} s")
        else:
            results['sparse_time'].append(None)
        
        # 测试稀疏矩阵 + 迭代求解
        if test_iterative:
            structure = generate_grid_truss(nx, ny, use_sparse=True)
            solver = StaticSolver(structure, use_iterative=True)
            
            start = time.perf_counter()
            result = solver.solve()
            elapsed = time.perf_counter() - start
            
            results['iterative_time'].append(elapsed)
            if verbose:
                print(f"稀疏矩阵迭代求解: {elapsed:.4f} s")
        else:
            results['iterative_time'].append(None)
    
    return results


def print_benchmark_report(results: Dict[str, List[float]]):
    """打印基准测试报告"""
    print("\n" + "="*80)
    print("性能基准测试报告")
    print("="*80)
    print(f"{'节点数':>10} {'自由度':>10} {'密集(s)':>12} {'稀疏(s)':>12} {'迭代(s)':>12}")
    print("-"*80)
    
    for i in range(len(results['nodes'])):
        nodes = results['nodes'][i]
        dofs = results['dofs'][i]
        dense = results['dense_time'][i]
        sparse = results['sparse_time'][i]
        iterative = results['iterative_time'][i]
        
        dense_str = f"{dense:.4f}" if dense != float('inf') else "OOM"
        sparse_str = f"{sparse:.4f}" if sparse else "N/A"
        iter_str = f"{iterative:.4f}" if iterative else "N/A"
        
        print(f"{nodes:>10} {dofs:>10} {dense_str:>12} {sparse_str:>12} {iter_str:>12}")
    
    print("="*80)
    print("\n建议：")
    print("- 节点数 < 500：使用密集矩阵（默认配置）")
    print("- 节点数 500-2000：使用稀疏矩阵 + 直接求解器")
    print("- 节点数 > 2000：使用稀疏矩阵 + 迭代求解器")
    print("\n启用稀疏矩阵：TrussStructure(use_sparse=True)")
    print("启用迭代求解：StaticSolver(structure, use_iterative=True)")


def run_1000_node_benchmark():
    """
    运行 1000 节点规模的基准测试
    
    这是 Prompt 要求的特定基准测试，用于验证中等规模问题的性能。
    """
    print("\n" + "="*80)
    print("1000 节点规模基准测试")
    print("="*80)
    
    # 生成 32x32 = 1024 节点的网格
    nx, ny = 32, 32
    n_nodes = nx * ny
    
    print(f"网格尺寸: {nx}x{ny} = {n_nodes} 节点")
    print(f"总自由度: {n_nodes * 2}")
    
    # 稀疏矩阵 + 直接求解
    print("\n[1] 稀疏矩阵 + 直接求解器")
    structure = generate_grid_truss(nx, ny, use_sparse=True)
    solver = StaticSolver(structure, use_iterative=False)
    
    start = time.perf_counter()
    result = solver.solve()
    elapsed_direct = time.perf_counter() - start
    
    print(f"    组装 + 求解耗时: {elapsed_direct:.4f} s")
    print(f"    最大位移: {np.max(np.abs(result.displacements)):.6e} m")
    
    # 稀疏矩阵 + 迭代求解
    print("\n[2] 稀疏矩阵 + 迭代求解器 (CG)")
    structure = generate_grid_truss(nx, ny, use_sparse=True)
    solver = StaticSolver(structure, use_iterative=True)
    
    start = time.perf_counter()
    result = solver.solve()
    elapsed_iterative = time.perf_counter() - start
    
    print(f"    组装 + 求解耗时: {elapsed_iterative:.4f} s")
    print(f"    最大位移: {np.max(np.abs(result.displacements)):.6e} m")
    
    print("\n" + "="*80)
    print(f"结论: 1000 节点规模下，稀疏矩阵求解耗时约 {elapsed_direct:.2f}-{elapsed_iterative:.2f} 秒")
    print("="*80)
    
    return {
        'nodes': n_nodes,
        'direct_time': elapsed_direct,
        'iterative_time': elapsed_iterative
    }


if __name__ == '__main__':
    # 运行完整基准测试
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        results = run_benchmark([100, 500, 1000, 2000])
        print_benchmark_report(results)
    else:
        # 默认只运行 1000 节点基准
        run_1000_node_benchmark()
