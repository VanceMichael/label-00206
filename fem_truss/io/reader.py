"""数据读取模块"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from ..core import TrussStructure, Material
from ..utils import get_logger, InputError, ValidationError

logger = get_logger(__name__)


class DataReader:
    """
    数据文件读取器
    
    支持读取节点、单元、材料、荷载、地震波等CSV格式数据文件
    """
    
    @staticmethod
    def read_nodes(filepath: str) -> pd.DataFrame:
        """
        读取节点数据文件
        
        文件格式: id, x, y
        
        Args:
            filepath: CSV文件路径
            
        Returns:
            节点数据DataFrame
        """
        logger.info(f"读取节点数据: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise InputError(f"节点数据文件不存在: {filepath}")
        except Exception as e:
            raise InputError(f"读取节点数据失败: {e}")
        
        # 验证必要列
        required_cols = ['id', 'x', 'y']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValidationError(f"节点数据缺少必要列: {missing}")
        
        # 数据验证
        if df['id'].duplicated().any():
            raise ValidationError("节点ID存在重复")
        
        logger.info(f"成功读取 {len(df)} 个节点")
        return df
    
    @staticmethod
    def read_elements(filepath: str) -> pd.DataFrame:
        """
        读取单元数据文件
        
        文件格式: id, node_i, node_j, material_id, area
        
        Args:
            filepath: CSV文件路径
            
        Returns:
            单元数据DataFrame
        """
        logger.info(f"读取单元数据: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise InputError(f"单元数据文件不存在: {filepath}")
        except Exception as e:
            raise InputError(f"读取单元数据失败: {e}")
        
        required_cols = ['id', 'node_i', 'node_j', 'material_id', 'area']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValidationError(f"单元数据缺少必要列: {missing}")
        
        # 数据验证
        if df['id'].duplicated().any():
            raise ValidationError("单元ID存在重复")
        if (df['area'] <= 0).any():
            raise ValidationError("截面面积必须为正值")
        
        logger.info(f"成功读取 {len(df)} 个单元")
        return df
    
    @staticmethod
    def read_materials(filepath: str) -> pd.DataFrame:
        """
        读取材料数据文件
        
        文件格式: id, name, E, nu, rho
        
        Args:
            filepath: CSV文件路径
            
        Returns:
            材料数据DataFrame
        """
        logger.info(f"读取材料数据: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise InputError(f"材料数据文件不存在: {filepath}")
        except Exception as e:
            raise InputError(f"读取材料数据失败: {e}")
        
        required_cols = ['id', 'name', 'E']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValidationError(f"材料数据缺少必要列: {missing}")
        
        # 设置默认值
        if 'nu' not in df.columns:
            df['nu'] = 0.3
        if 'rho' not in df.columns:
            df['rho'] = 7850.0
        
        # 数据验证
        if (df['E'] <= 0).any():
            raise ValidationError("弹性模量必须为正值")
        
        logger.info(f"成功读取 {len(df)} 种材料")
        return df
    
    @staticmethod
    def read_loads(filepath: str) -> pd.DataFrame:
        """
        读取荷载数据文件
        
        文件格式: id, node_id, fx, fy
        
        Args:
            filepath: CSV文件路径
            
        Returns:
            荷载数据DataFrame
        """
        logger.info(f"读取荷载数据: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise InputError(f"荷载数据文件不存在: {filepath}")
        except Exception as e:
            raise InputError(f"读取荷载数据失败: {e}")
        
        required_cols = ['node_id']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValidationError(f"荷载数据缺少必要列: {missing}")
        
        # 设置默认值
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
        if 'fx' not in df.columns:
            df['fx'] = 0.0
        if 'fy' not in df.columns:
            df['fy'] = 0.0
        
        logger.info(f"成功读取 {len(df)} 个荷载")
        return df
    
    @staticmethod
    def read_boundaries(filepath: str) -> pd.DataFrame:
        """
        读取边界条件数据文件
        
        文件格式: node_id, fix_x, fix_y
        
        Args:
            filepath: CSV文件路径
            
        Returns:
            边界条件数据DataFrame
        """
        logger.info(f"读取边界条件数据: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise InputError(f"边界条件数据文件不存在: {filepath}")
        except Exception as e:
            raise InputError(f"读取边界条件数据失败: {e}")
        
        required_cols = ['node_id']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValidationError(f"边界条件数据缺少必要列: {missing}")
        
        # 设置默认值
        if 'fix_x' not in df.columns:
            df['fix_x'] = True
        if 'fix_y' not in df.columns:
            df['fix_y'] = True
        
        # 转换布尔值
        df['fix_x'] = df['fix_x'].astype(bool)
        df['fix_y'] = df['fix_y'].astype(bool)
        
        logger.info(f"成功读取 {len(df)} 个边界条件")
        return df
    
    @staticmethod
    def read_seismic(filepath: str) -> pd.DataFrame:
        """
        读取地震波加速度时程数据
        
        文件格式: time, acceleration
        
        Args:
            filepath: CSV文件路径
            
        Returns:
            地震波数据DataFrame
        """
        logger.info(f"读取地震波数据: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise InputError(f"地震波数据文件不存在: {filepath}")
        except Exception as e:
            raise InputError(f"读取地震波数据失败: {e}")
        
        required_cols = ['time', 'acceleration']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValidationError(f"地震波数据缺少必要列: {missing}")
        
        # 验证时间序列
        if not df['time'].is_monotonic_increasing:
            logger.warning("地震波时间序列非单调递增，将进行排序")
            df = df.sort_values('time').reset_index(drop=True)
        
        logger.info(f"成功读取地震波数据: {len(df)} 个时间点, "
                   f"时长 {df['time'].max():.2f}s")
        return df
    
    @classmethod
    def load_structure(cls, data_dir: str) -> TrussStructure:
        """
        从数据目录加载完整的桁架结构
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            构建好的TrussStructure对象
        """
        data_path = Path(data_dir)
        logger.info(f"从目录加载结构: {data_path}")
        
        structure = TrussStructure()
        
        # 读取材料
        materials_file = data_path / 'materials.csv'
        if materials_file.exists():
            materials_df = cls.read_materials(str(materials_file))
            for _, row in materials_df.iterrows():
                material = Material(
                    id=int(row['id']),
                    name=row['name'],
                    E=float(row['E']),
                    nu=float(row.get('nu', 0.3)),
                    rho=float(row.get('rho', 7850))
                )
                structure.add_material(material)
        else:
            # 添加默认材料
            structure.add_material(Material(id=1, name="Steel", E=2.06e11))
        
        # 读取节点
        nodes_df = cls.read_nodes(str(data_path / 'nodes.csv'))
        for _, row in nodes_df.iterrows():
            structure.add_node(int(row['id']), float(row['x']), float(row['y']))
        
        # 读取单元
        elements_df = cls.read_elements(str(data_path / 'elements.csv'))
        for _, row in elements_df.iterrows():
            structure.add_element(
                id=int(row['id']),
                node_i=int(row['node_i']),
                node_j=int(row['node_j']),
                material_id=int(row['material_id']),
                area=float(row['area'])
            )
        
        # 读取荷载
        loads_file = data_path / 'loads.csv'
        if loads_file.exists():
            loads_df = cls.read_loads(str(loads_file))
            for _, row in loads_df.iterrows():
                structure.apply_load(
                    node_id=int(row['node_id']),
                    fx=float(row.get('fx', 0)),
                    fy=float(row.get('fy', 0))
                )
        
        # 读取边界条件
        boundaries_file = data_path / 'boundaries.csv'
        if boundaries_file.exists():
            boundaries_df = cls.read_boundaries(str(boundaries_file))
            for _, row in boundaries_df.iterrows():
                structure.apply_boundary(
                    node_id=int(row['node_id']),
                    fix_x=bool(row.get('fix_x', True)),
                    fix_y=bool(row.get('fix_y', True))
                )
        
        logger.info(f"结构加载完成:\n{structure.summary()}")
        return structure
