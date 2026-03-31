"""结果输出模块"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class StaticResult:
    """静力分析结果"""
    displacements: np.ndarray  # 节点位移
    stresses: Dict[int, float]  # 单元应力
    strains: Dict[int, float]   # 单元应变
    reactions: np.ndarray       # 支座反力


@dataclass
class DynamicResult:
    """动力分析结果"""
    time: np.ndarray                    # 时间序列
    displacement_history: np.ndarray    # 位移时程 (n_steps, ndof)
    velocity_history: np.ndarray        # 速度时程
    acceleration_history: np.ndarray    # 加速度时程


class ResultWriter:
    """
    结果输出器
    
    将分析结果输出为CSV文件
    """
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"结果输出目录: {self.output_dir}")
    
    def write_displacements(self, displacements: np.ndarray, 
                           node_ids: List[int],
                           filename: str = 'displacements.csv'):
        """
        输出节点位移结果
        
        Args:
            displacements: 位移向量
            node_ids: 节点ID列表
            filename: 输出文件名
        """
        filepath = self.output_dir / filename
        
        data = []
        for i, node_id in enumerate(node_ids):
            data.append({
                'node_id': node_id,
                'ux (m)': displacements[2*i],
                'uy (m)': displacements[2*i + 1],
                'magnitude (m)': np.sqrt(displacements[2*i]**2 + displacements[2*i+1]**2)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, float_format='%.6e')
        
        logger.info(f"位移结果已保存: {filepath}")
        return filepath
    
    def write_stresses(self, stresses: Dict[int, float],
                      strains: Optional[Dict[int, float]] = None,
                      filename: str = 'stresses.csv'):
        """
        输出单元应力结果
        
        Args:
            stresses: 单元应力字典 {element_id: stress}
            strains: 单元应变字典（可选）
            filename: 输出文件名
        """
        filepath = self.output_dir / filename
        
        data = []
        for elem_id, stress in stresses.items():
            row = {
                'element_id': elem_id,
                'stress (Pa)': stress,
                'stress (MPa)': stress / 1e6
            }
            if strains and elem_id in strains:
                row['strain'] = strains[elem_id]
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, float_format='%.6e')
        
        logger.info(f"应力结果已保存: {filepath}")
        return filepath
    
    def write_reactions(self, reactions: np.ndarray,
                       constrained_dofs: List[int],
                       node_ids: List[int],
                       filename: str = 'reactions.csv'):
        """
        输出支座反力结果
        
        Args:
            reactions: 反力向量
            constrained_dofs: 约束自由度列表
            node_ids: 节点ID列表
            filename: 输出文件名
        """
        filepath = self.output_dir / filename
        
        data = []
        for dof in constrained_dofs:
            node_idx = dof // 2
            direction = 'x' if dof % 2 == 0 else 'y'
            data.append({
                'node_id': node_ids[node_idx],
                'direction': direction,
                'reaction (N)': reactions[dof]
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, float_format='%.6e')
        
        logger.info(f"反力结果已保存: {filepath}")
        return filepath
    
    def write_time_history(self, result: DynamicResult,
                          node_indices: List[int],
                          filename: str = 'time_history.csv'):
        """
        输出时程分析结果
        
        Args:
            result: 动力分析结果
            node_indices: 要输出的节点索引
            filename: 输出文件名
        """
        filepath = self.output_dir / filename
        
        data = {'time (s)': result.time}
        
        for idx in node_indices:
            dof_x = 2 * idx
            dof_y = 2 * idx + 1
            data[f'node_{idx}_ux (m)'] = result.displacement_history[:, dof_x]
            data[f'node_{idx}_uy (m)'] = result.displacement_history[:, dof_y]
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, float_format='%.6e')
        
        logger.info(f"时程结果已保存: {filepath}")
        return filepath
    
    def write_static_result(self, result: StaticResult,
                           node_ids: List[int],
                           constrained_dofs: List[int]):
        """
        输出完整的静力分析结果
        
        Args:
            result: 静力分析结果
            node_ids: 节点ID列表
            constrained_dofs: 约束自由度列表
        """
        self.write_displacements(result.displacements, node_ids)
        self.write_stresses(result.stresses, result.strains)
        self.write_reactions(result.reactions, constrained_dofs, node_ids)
        
        logger.info("静力分析结果输出完成")
