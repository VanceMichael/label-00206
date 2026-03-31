"""可视化绘图模块

本模块提供桁架结构的可视化功能，包括：
- 结构几何图绘制
- 变形放大图绘制
- 应力云图绘制（支持 3D 曲面插值）
- 动力响应动画生成
- 时程曲线绘制

Notebook 交互式使用说明：
在 Jupyter Notebook 中使用交互式绘图功能，推荐使用以下设置：

    %matplotlib notebook  # 启用交互式后端，支持缩放、平移等操作
    
    # 或使用 widget 后端（需安装 ipympl）
    %matplotlib widget
    
    from fem_truss.visualization import TrussPlotter
    plotter = TrussPlotter(structure, output_dir='output')
    fig = plotter.plot_structure(save=False)  # 设置 save=False 以在 notebook 中显示
    plt.show()

交互式模式的优势：
- 支持鼠标缩放和平移
- 可实时调整视图
- 适合探索性数据分析

非交互式模式（默认）：
- 适用于脚本批量处理
- 适用于 Web 服务器环境
- 自动保存图片到文件
"""
import os
import threading
import numpy as np
import matplotlib

# 智能选择 matplotlib 后端
# 规则：
# 1. 如果显式设置了 MPLBACKEND=Agg 或 FEM_TRUSS_WEB_MODE，使用 Agg
# 2. 如果不在主线程中运行（如 Flask 工作线程），必须使用 Agg（macOS 限制）
# 3. 如果当前后端是 GUI 后端（MacOSX, TkAgg 等）且无显示环境，使用 Agg
# 4. 其他情况保持默认后端（支持 Notebook 交互式环境）

def _should_use_agg_backend() -> bool:
    """判断是否应该使用 Agg 非交互式后端"""
    # 显式指定使用 Agg
    if os.environ.get('MPLBACKEND') == 'Agg':
        return True
    if os.environ.get('FEM_TRUSS_WEB_MODE'):
        return True
    
    # 非主线程必须使用 Agg（macOS 的 NSWindow 限制）
    if threading.current_thread() is not threading.main_thread():
        return True
    
    # 无显示环境且使用 GUI 后端
    gui_backends = ['MacOSX', 'TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'WXAgg']
    current_backend = matplotlib.get_backend()
    if not os.environ.get('DISPLAY') and current_backend in gui_backends:
        return True
    
    return False

if _should_use_agg_backend():
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..core import TrussStructure
from ..io.writer import DynamicResult
from ..utils import get_logger

logger = get_logger(__name__)

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TrussPlotter:
    """
    桁架结构可视化绘图器
    
    提供结构图、变形图、应力云图、动画等可视化功能。
    
    使用示例：
        # 基本用法
        plotter = TrussPlotter(structure, output_dir='output')
        plotter.plot_structure()
        plotter.plot_deformation(result.displacements)
        plotter.plot_stress_contour(result.stresses)
        
        # Notebook 交互式用法
        %matplotlib notebook
        fig = plotter.plot_structure(save=False)
        plt.show()
    """
    
    def __init__(self, structure: TrussStructure, output_dir: str = 'output'):
        """
        初始化绘图器
        
        Args:
            structure: 桁架结构对象
            output_dir: 输出目录
        """
        self.structure = structure
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 颜色配置
        self.colors = {
            'node': '#1a365d',
            'element': '#3182ce',
            'deformed': '#e53e3e',
            'load': '#38a169',
            'boundary': '#dd6b20',
            'background': '#f7fafc'
        }
        
    def _get_node_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取所有节点坐标"""
        sorted_nodes = sorted(self.structure.nodes.values(), key=lambda n: n.id)
        x = np.array([n.x for n in sorted_nodes])
        y = np.array([n.y for n in sorted_nodes])
        return x, y
    
    def _get_element_lines(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """获取所有单元线段"""
        lines = []
        for elem in self.structure.elements.values():
            ni = self.structure.nodes[elem.node_i]
            nj = self.structure.nodes[elem.node_j]
            lines.append(((ni.x, ni.y), (nj.x, nj.y)))
        return lines
    
    def plot_structure(self, show_labels: bool = True,
                      figsize: Tuple[int, int] = (10, 8),
                      save: bool = True) -> plt.Figure:
        """
        绘制原始结构几何图
        
        Args:
            show_labels: 是否显示节点和单元编号
            figsize: 图形尺寸
            save: 是否保存图片（Notebook 交互模式建议设为 False）
            
        Returns:
            Figure 对象
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        
        # 绘制单元
        lines = self._get_element_lines()
        lc = LineCollection(lines, colors=self.colors['element'], linewidths=2)
        ax.add_collection(lc)
        
        # 绘制节点
        x, y = self._get_node_coords()
        ax.scatter(x, y, c=self.colors['node'], s=100, zorder=5)
        
        # 绘制边界条件
        for bc in self.structure.boundaries:
            node = self.structure.nodes[bc.node_id]
            marker_size = 200
            if bc.fix_x and bc.fix_y:
                ax.scatter(node.x, node.y, marker='^', c=self.colors['boundary'],
                          s=marker_size, zorder=6, label='Fixed' if bc.id == 1 else '')
            elif bc.fix_x:
                ax.scatter(node.x, node.y, marker='>', c=self.colors['boundary'],
                          s=marker_size, zorder=6)
            elif bc.fix_y:
                ax.scatter(node.x, node.y, marker='^', c=self.colors['boundary'],
                          s=marker_size, zorder=6)
        
        # 绘制荷载
        for load in self.structure.loads:
            node = self.structure.nodes[load.node_id]
            scale = 0.1 * max(max(x) - min(x), max(y) - min(y))
            if abs(load.fx) > 1e-10:
                direction = 1 if load.fx > 0 else -1
                ax.annotate('', xy=(node.x + direction * scale, node.y),
                           xytext=(node.x, node.y),
                           arrowprops=dict(arrowstyle='->', color=self.colors['load'], lw=2))
            if abs(load.fy) > 1e-10:
                direction = 1 if load.fy > 0 else -1
                ax.annotate('', xy=(node.x, node.y + direction * scale),
                           xytext=(node.x, node.y),
                           arrowprops=dict(arrowstyle='->', color=self.colors['load'], lw=2))
        
        # 添加标签
        if show_labels:
            for node in self.structure.nodes.values():
                ax.annotate(f'N{node.id}', (node.x, node.y),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, color=self.colors['node'])
            
            for elem in self.structure.elements.values():
                ni = self.structure.nodes[elem.node_i]
                nj = self.structure.nodes[elem.node_j]
                mid_x = (ni.x + nj.x) / 2
                mid_y = (ni.y + nj.y) / 2
                ax.annotate(f'E{elem.id}', (mid_x, mid_y),
                           fontsize=8, color=self.colors['element'],
                           ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 设置坐标轴
        margin = 0.1 * max(max(x) - min(x), max(y) - min(y))
        ax.set_xlim(min(x) - margin, max(x) + margin)
        ax.set_ylim(min(y) - margin, max(y) + margin)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Truss Structure', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'structure.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight',
                       facecolor=self.colors['background'])
            logger.info(f"结构图已保存: {filepath}")
        
        plt.close(fig)
        return fig
    
    def plot_deformation(self, displacements: np.ndarray,
                        scale: float = 1.0,
                        auto_scale: bool = True,
                        figsize: Tuple[int, int] = (12, 8),
                        save: bool = True) -> plt.Figure:
        """
        绘制变形放大图
        
        Args:
            displacements: 节点位移向量
            scale: 变形放大系数
            auto_scale: 是否自动计算放大系数
            figsize: 图形尺寸
            save: 是否保存图片
            
        Returns:
            Figure 对象
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        
        # 获取原始坐标
        sorted_nodes = sorted(self.structure.nodes.values(), key=lambda n: n.id)
        x_orig = np.array([n.x for n in sorted_nodes])
        y_orig = np.array([n.y for n in sorted_nodes])
        
        # 计算变形后坐标
        if auto_scale:
            max_disp = np.max(np.abs(displacements))
            if max_disp > 1e-10:
                char_length = max(max(x_orig) - min(x_orig), max(y_orig) - min(y_orig))
                scale = 0.1 * char_length / max_disp
            logger.info(f"自动放大系数: {scale:.2f}")
        
        x_def = x_orig + scale * displacements[0::2]
        y_def = y_orig + scale * displacements[1::2]
        
        # 绘制原始结构（虚线）
        for elem in self.structure.elements.values():
            i = list(self.structure.nodes.keys()).index(elem.node_i)
            j = list(self.structure.nodes.keys()).index(elem.node_j)
            ax.plot([x_orig[i], x_orig[j]], [y_orig[i], y_orig[j]],
                   '--', color=self.colors['element'], alpha=0.5, linewidth=1.5)
        
        # 绘制变形后结构（实线）
        for elem in self.structure.elements.values():
            i = list(self.structure.nodes.keys()).index(elem.node_i)
            j = list(self.structure.nodes.keys()).index(elem.node_j)
            ax.plot([x_def[i], x_def[j]], [y_def[i], y_def[j]],
                   '-', color=self.colors['deformed'], linewidth=2)
        
        # 绘制节点
        ax.scatter(x_orig, y_orig, c=self.colors['element'], s=60, alpha=0.5,
                  label='Original')
        ax.scatter(x_def, y_def, c=self.colors['deformed'], s=80,
                  label='Deformed')
        
        # 设置坐标轴
        all_x = np.concatenate([x_orig, x_def])
        all_y = np.concatenate([y_orig, y_def])
        margin = 0.15 * max(max(all_x) - min(all_x), max(all_y) - min(all_y))
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'Deformation (Scale: {scale:.1f}x)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'deformation.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight',
                       facecolor=self.colors['background'])
            logger.info(f"变形图已保存: {filepath}")
        
        plt.close(fig)
        return fig
    
    def plot_stress_contour(self, stresses: Dict[int, float],
                           figsize: Tuple[int, int] = (12, 8),
                           save: bool = True,
                           use_3d_interpolation: bool = True) -> plt.Figure:
        """
        绘制应力云图（支持3D曲面插值）
        
        基于 Prompt 要求，采用3D曲面插值技术展示应力分布。
        
        Args:
            stresses: 单元应力字典
            figsize: 图形尺寸
            save: 是否保存图片
            use_3d_interpolation: 是否使用3D曲面插值
            
        Returns:
            Figure 对象
        """
        from scipy.interpolate import griddata
        
        # 准备数据
        stress_values = list(stresses.values())
        stress_min = min(stress_values)
        stress_max = max(stress_values)
        
        # 归一化
        norm = Normalize(vmin=stress_min, vmax=stress_max)
        cmap = plt.cm.RdYlBu_r  # 红-黄-蓝 颜色映射
        
        x, y = self._get_node_coords()
        
        if use_3d_interpolation and len(self.structure.elements) >= 3:
            # 使用3D曲面插值
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=figsize, facecolor=self.colors['background'])
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor(self.colors['background'])
            
            # 计算单元中心点的应力值
            elem_centers_x = []
            elem_centers_y = []
            elem_stresses = []
            
            for elem_id, elem in self.structure.elements.items():
                ni = self.structure.nodes[elem.node_i]
                nj = self.structure.nodes[elem.node_j]
                elem_centers_x.append((ni.x + nj.x) / 2)
                elem_centers_y.append((ni.y + nj.y) / 2)
                elem_stresses.append(stresses[elem_id])
            
            elem_centers_x = np.array(elem_centers_x)
            elem_centers_y = np.array(elem_centers_y)
            elem_stresses = np.array(elem_stresses)
            
            # 创建插值网格
            margin = 0.1 * max(max(x) - min(x), max(y) - min(y))
            xi = np.linspace(min(x) - margin, max(x) + margin, 50)
            yi = np.linspace(min(y) - margin, max(y) + margin, 50)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # 3D曲面插值
            try:
                Zi = griddata((elem_centers_x, elem_centers_y), elem_stresses,
                             (Xi, Yi), method='cubic', fill_value=np.nan)
                
                # 绘制3D曲面
                surf = ax.plot_surface(Xi, Yi, Zi, cmap=cmap, norm=norm,
                                       alpha=0.8, linewidth=0, antialiased=True)
                
                # 绘制单元线框
                for elem in self.structure.elements.values():
                    ni = self.structure.nodes[elem.node_i]
                    nj = self.structure.nodes[elem.node_j]
                    stress = stresses[elem.id]
                    z_val = stress
                    ax.plot([ni.x, nj.x], [ni.y, nj.y], [z_val, z_val],
                           'k-', linewidth=1.5, alpha=0.7)
                
                # 添加颜色条
                cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
                cbar.set_label('Stress (Pa)', fontsize=12)
                
                ax.set_xlabel('X (m)', fontsize=10)
                ax.set_ylabel('Y (m)', fontsize=10)
                ax.set_zlabel('Stress (Pa)', fontsize=10)
                ax.set_title('3D Stress Distribution (Interpolated Surface)', 
                            fontsize=14, fontweight='bold')
                
            except Exception as e:
                logger.warning(f"3D插值失败，回退到2D显示: {e}")
                plt.close(fig)
                return self._plot_stress_contour_2d(stresses, figsize, save)
        else:
            # 2D 应力云图
            return self._plot_stress_contour_2d(stresses, figsize, save)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'stress_contour.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight',
                       facecolor=self.colors['background'])
            logger.info(f"应力云图已保存: {filepath}")
        
        plt.close(fig)
        return fig
    
    def _plot_stress_contour_2d(self, stresses: Dict[int, float],
                                figsize: Tuple[int, int] = (12, 8),
                                save: bool = True) -> plt.Figure:
        """
        绘制2D应力云图（备选方案）
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        
        stress_values = list(stresses.values())
        stress_min = min(stress_values)
        stress_max = max(stress_values)
        
        norm = Normalize(vmin=stress_min, vmax=stress_max)
        cmap = plt.cm.RdYlBu_r
        
        # 绘制单元（按应力着色，使用渐变线宽）
        lines = []
        colors = []
        linewidths = []
        
        for elem_id, elem in self.structure.elements.items():
            ni = self.structure.nodes[elem.node_i]
            nj = self.structure.nodes[elem.node_j]
            lines.append(((ni.x, ni.y), (nj.x, nj.y)))
            colors.append(cmap(norm(stresses[elem_id])))
            # 线宽与应力绝对值成正比
            lw = 2 + 4 * abs(norm(stresses[elem_id]) - 0.5)
            linewidths.append(lw)
        
        lc = LineCollection(lines, colors=colors, linewidths=linewidths)
        ax.add_collection(lc)
        
        # 绘制节点
        x, y = self._get_node_coords()
        ax.scatter(x, y, c='black', s=50, zorder=5)
        
        # 添加应力值标注
        for elem_id, elem in self.structure.elements.items():
            ni = self.structure.nodes[elem.node_i]
            nj = self.structure.nodes[elem.node_j]
            mid_x = (ni.x + nj.x) / 2
            mid_y = (ni.y + nj.y) / 2
            stress_mpa = stresses[elem_id] / 1e6
            ax.annotate(f'{stress_mpa:.2f}', (mid_x, mid_y),
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 添加颜色条
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Stress (Pa)', fontsize=12)
        
        # 设置坐标轴
        margin = 0.15 * max(max(x) - min(x), max(y) - min(y))
        ax.set_xlim(min(x) - margin, max(x) + margin)
        ax.set_ylim(min(y) - margin, max(y) + margin)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Stress Distribution (MPa values shown)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'stress_contour.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight',
                       facecolor=self.colors['background'])
            logger.info(f"应力云图已保存: {filepath}")
        
        plt.close(fig)
        return fig
    
    def create_animation(self, result: DynamicResult,
                        scale: float = None,
                        interval: int = 50,
                        save: bool = True,
                        filename: str = 'animation.gif') -> FuncAnimation:
        """
        创建位移时程动画
        
        Args:
            result: 动力分析结果
            scale: 变形放大系数
            interval: 帧间隔 (ms)
            save: 是否保存动画
            filename: 输出文件名
            
        Returns:
            FuncAnimation 对象
        """
        logger.info("创建动画...")
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        
        # 获取原始坐标
        sorted_nodes = sorted(self.structure.nodes.values(), key=lambda n: n.id)
        x_orig = np.array([n.x for n in sorted_nodes])
        y_orig = np.array([n.y for n in sorted_nodes])
        
        # 自动计算放大系数
        if scale is None:
            max_disp = np.max(np.abs(result.displacement_history))
            if max_disp > 1e-10:
                char_length = max(max(x_orig) - min(x_orig), max(y_orig) - min(y_orig))
                scale = 0.1 * char_length / max_disp
        
        # 绘制原始结构
        for elem in self.structure.elements.values():
            i = list(self.structure.nodes.keys()).index(elem.node_i)
            j = list(self.structure.nodes.keys()).index(elem.node_j)
            ax.plot([x_orig[i], x_orig[j]], [y_orig[i], y_orig[j]],
                   '--', color=self.colors['element'], alpha=0.3, linewidth=1)
        
        # 初始化动画元素
        lines = []
        for elem in self.structure.elements.values():
            line, = ax.plot([], [], '-', color=self.colors['deformed'], linewidth=2)
            lines.append(line)
        
        scatter = ax.scatter([], [], c=self.colors['deformed'], s=60)
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 设置坐标轴范围
        max_disp_scaled = np.max(np.abs(result.displacement_history)) * scale
        margin = max(0.1 * max(max(x_orig) - min(x_orig), max(y_orig) - min(y_orig)),
                    max_disp_scaled * 1.5)
        ax.set_xlim(min(x_orig) - margin, max(x_orig) + margin)
        ax.set_ylim(min(y_orig) - margin, max(y_orig) + margin)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Dynamic Response Animation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 采样帧（减少帧数以控制文件大小）
        n_frames = min(200, len(result.time))
        frame_indices = np.linspace(0, len(result.time) - 1, n_frames, dtype=int)
        
        def init():
            for line in lines:
                line.set_data([], [])
            scatter.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            return lines + [scatter, time_text]
        
        def animate(frame_idx):
            idx = frame_indices[frame_idx]
            disp = result.displacement_history[idx]
            
            x_def = x_orig + scale * disp[0::2]
            y_def = y_orig + scale * disp[1::2]
            
            # 更新单元线段
            for i, elem in enumerate(self.structure.elements.values()):
                ni_idx = list(self.structure.nodes.keys()).index(elem.node_i)
                nj_idx = list(self.structure.nodes.keys()).index(elem.node_j)
                lines[i].set_data([x_def[ni_idx], x_def[nj_idx]],
                                 [y_def[ni_idx], y_def[nj_idx]])
            
            # 更新节点
            scatter.set_offsets(np.column_stack([x_def, y_def]))
            
            # 更新时间文本
            time_text.set_text(f't = {result.time[idx]:.3f} s')
            
            return lines + [scatter, time_text]
        
        anim = FuncAnimation(fig, animate, init_func=init,
                            frames=n_frames, interval=interval, blit=True)
        
        if save:
            filepath = self.output_dir / filename
            logger.info(f"保存动画到: {filepath}")
            anim.save(str(filepath), writer='pillow', fps=20)
            logger.info(f"动画已保存: {filepath}")
        
        plt.close(fig)
        return anim
    
    def plot_time_history(self, result: DynamicResult,
                         node_indices: List[int],
                         direction: str = 'y',
                         figsize: Tuple[int, int] = (12, 6),
                         save: bool = True) -> plt.Figure:
        """
        绘制节点位移时程曲线
        
        Args:
            result: 动力分析结果
            node_indices: 要绘制的节点索引列表
            direction: 位移方向 ('x' 或 'y')
            figsize: 图形尺寸
            save: 是否保存图片
            
        Returns:
            Figure 对象
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(node_indices)))
        
        for i, node_idx in enumerate(node_indices):
            dof = 2 * node_idx + (0 if direction == 'x' else 1)
            disp = result.displacement_history[:, dof]
            ax.plot(result.time, disp * 1000, color=colors[i],
                   label=f'Node {node_idx + 1}', linewidth=1.5)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(f'Displacement {direction.upper()} (mm)', fontsize=12)
        ax.set_title('Displacement Time History', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'time_history.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight',
                       facecolor=self.colors['background'])
            logger.info(f"时程曲线已保存: {filepath}")
        
        plt.close(fig)
        return fig
