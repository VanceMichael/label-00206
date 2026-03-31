"""桁架单元类定义

理论基础：
本模块实现平面桁架单元的有限元分析。

关于 CST 理论与桁架单元的关系：
- CST（Constant Strain Triangle）是一种平面应力/应变三角形单元，用于2D连续体分析
- 对于桁架结构，CST 理论简化为轴向拉压杆单元理论（Bar/Truss Element）
- 桁架单元是 CST 单元的一维退化形式，仅承受轴向力，不承受弯矩和剪力

桁架单元的基本假设：
1. 单元仅承受轴向力（拉力或压力）
2. 单元两端为铰接，不传递弯矩
3. 单元内应变为常数（与 CST 的常应变假设一致）
4. 材料为线弹性，满足胡克定律 σ = E·ε

单元刚度矩阵推导：
- 应变-位移关系：ε = (u_j - u_i) / L = B·u，其中 B = [-c/L, -s/L, c/L, s/L]
- 本构关系：σ = E·ε（单轴应力状态）
- 刚度矩阵：K = ∫ B^T·E·B·A dL = (EA/L)·T^T·k_local·T

参考文献：
- Zienkiewicz, O.C., Taylor, R.L. "The Finite Element Method"
- Cook, R.D. "Concepts and Applications of Finite Element Analysis"
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .material import Material

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrussElement:
    """
    平面桁架单元类（轴向拉压杆单元）
    
    本类实现二维平面内的桁架单元，基于轴向拉压杆理论。
    该理论是 CST 平面应力理论在一维杆件中的简化形式。
    
    Attributes:
        id: 单元编号
        node_i: 起始节点编号
        node_j: 终止节点编号
        material: 材料属性
        area: 截面面积 (m²)
        coords_i: 起始节点坐标 (x, y)
        coords_j: 终止节点坐标 (x, y)
    """
    id: int
    node_i: int
    node_j: int
    material: 'Material'
    area: float
    coords_i: Tuple[float, float] = field(default=(0.0, 0.0))
    coords_j: Tuple[float, float] = field(default=(0.0, 0.0))
    
    # 计算属性（延迟初始化）
    _length: float = field(init=False, repr=False, default=0.0)
    _cos_theta: float = field(init=False, repr=False, default=0.0)
    _sin_theta: float = field(init=False, repr=False, default=0.0)
    
    def __post_init__(self):
        """初始化几何属性"""
        if self.area <= 0:
            raise ValueError(f"截面面积必须为正值，当前值: {self.area}")
        self._compute_geometry()
    
    def _compute_geometry(self):
        """计算单元几何属性"""
        dx = self.coords_j[0] - self.coords_i[0]
        dy = self.coords_j[1] - self.coords_i[1]
        self._length = np.sqrt(dx**2 + dy**2)
        
        if self._length < 1e-10:
            raise ValueError(f"单元 {self.id} 长度过小: {self._length}")
        
        self._cos_theta = dx / self._length
        self._sin_theta = dy / self._length
    
    def update_coordinates(self, coords_i: Tuple[float, float], coords_j: Tuple[float, float]):
        """更新节点坐标并重新计算几何属性"""
        self.coords_i = coords_i
        self.coords_j = coords_j
        self._compute_geometry()
    
    @property
    def length(self) -> float:
        """单元长度"""
        return self._length
    
    @property
    def cos_theta(self) -> float:
        """方向余弦 cos(θ)"""
        return self._cos_theta
    
    @property
    def sin_theta(self) -> float:
        """方向正弦 sin(θ)"""
        return self._sin_theta
    
    @property
    def angle(self) -> float:
        """单元倾角（弧度）"""
        return np.arctan2(self._sin_theta, self._cos_theta)
    
    def _get_transformation_matrix(self) -> np.ndarray:
        """
        获取坐标变换矩阵 T
        
        将局部坐标系下的位移/力转换到整体坐标系
        
        Returns:
            2x4 变换矩阵
        """
        c = self._cos_theta
        s = self._sin_theta
        
        # 变换矩阵: 局部 -> 整体
        T = np.array([
            [c, s, 0, 0],
            [0, 0, c, s]
        ])
        return T
    
    def _get_strain_displacement_matrix(self) -> np.ndarray:
        """
        获取应变-位移矩阵 B（基于CST理论）
        
        对于桁架单元，B矩阵将节点位移映射到轴向应变：
        ε = B * u
        
        Returns:
            1x4 应变-位移矩阵
        """
        c = self._cos_theta
        s = self._sin_theta
        L = self._length
        
        # B = [-c/L, -s/L, c/L, s/L]
        B = np.array([[-c/L, -s/L, c/L, s/L]])
        return B
    
    def _get_constitutive_matrix(self) -> float:
        """
        获取本构矩阵 D
        
        理论说明：
        - 完整的平面应力本构矩阵为 3x3 矩阵：
          D = E/(1-ν²) * [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]
        
        - 对于桁架单元（轴向拉压杆），由于仅考虑单轴应力状态，
          本构关系简化为一维形式：σ = E·ε
          因此 D = E（弹性模量）
        
        这是 CST 理论在桁架单元中的简化应用。
        
        Returns:
            弹性模量 E
        """
        return self.material.E
    
    def compute_stiffness_matrix(self) -> np.ndarray:
        """
        计算单元刚度矩阵（整体坐标系）
        
        理论推导（轴向拉压杆单元）：
        
        1. 局部坐标系下的刚度矩阵（1D杆单元）：
           k_local = (EA/L) * [[1, -1], [-1, 1]]
        
        2. 坐标变换到整体坐标系：
           K = T^T · k_local · T
           
           其中变换矩阵 T 将整体坐标位移映射到局部轴向位移
        
        3. 最终形式：
           K = (EA/L) * [[c²,  cs, -c², -cs],
                         [cs,  s², -cs, -s²],
                         [-c², -cs, c²,  cs],
                         [-cs, -s², cs,  s²]]
           
           其中 c = cos(θ), s = sin(θ), θ 为单元倾角
        
        注：此公式与 CST 理论中 K = ∫B^T·D·B·dV 的结果一致，
        是 CST 理论在一维杆件中的简化形式。
        
        Returns:
            4x4 单元刚度矩阵
        """
        c = self._cos_theta
        s = self._sin_theta
        E = self.material.E
        A = self.area
        L = self._length
        
        # 基于CST理论的刚度系数
        k = E * A / L
        
        # 方向余弦的组合
        cc = c * c
        ss = s * s
        cs = c * s
        
        # 整体坐标系下的刚度矩阵
        # K = k * T^T * [[1,-1],[-1,1]] * T
        ke = k * np.array([
            [ cc,  cs, -cc, -cs],
            [ cs,  ss, -cs, -ss],
            [-cc, -cs,  cc,  cs],
            [-cs, -ss,  cs,  ss]
        ])
        
        logger.debug(f"单元 {self.id} 刚度矩阵计算完成, k={k:.2e}")
        return ke
    
    def compute_mass_matrix(self, consistent: bool = True) -> np.ndarray:
        """
        计算单元质量矩阵（整体坐标系）
        
        Args:
            consistent: True 使用一致质量矩阵，False 使用集中质量矩阵
            
        Returns:
            4x4 单元质量矩阵
        """
        m_total = self.material.rho * self.area * self._length
        
        if consistent:
            # 一致质量矩阵（基于形函数积分）
            # M = ρ * A * ∫ N^T * N dL
            c = self._cos_theta
            s = self._sin_theta
            
            # 局部坐标系下的一致质量矩阵
            m_local = (m_total / 6) * np.array([
                [2, 0, 1, 0],
                [0, 2, 0, 1],
                [1, 0, 2, 0],
                [0, 1, 0, 2]
            ])
            
            # 坐标变换矩阵
            T = np.array([
                [c, s, 0, 0],
                [-s, c, 0, 0],
                [0, 0, c, s],
                [0, 0, -s, c]
            ])
            
            me = T.T @ m_local @ T
        else:
            # 集中质量矩阵
            m_node = m_total / 2
            me = m_node * np.eye(4)
        
        logger.debug(f"单元 {self.id} 质量矩阵计算完成, m_total={m_total:.4f}")
        return me
    
    def compute_stress(self, displacements: np.ndarray) -> float:
        """
        计算单元应力（基于CST平面应力理论）
        
        σ = D * ε = E * B * u
        
        Args:
            displacements: 单元节点位移向量 [u_i, v_i, u_j, v_j]
            
        Returns:
            单元轴向应力 (正为拉应力，负为压应力)
        """
        if len(displacements) != 4:
            raise ValueError(f"位移向量长度必须为4，当前: {len(displacements)}")
        
        # 使用应变-位移矩阵计算应变
        B = self._get_strain_displacement_matrix()
        strain = (B @ displacements)[0]
        
        # 应力 = D * 应变 = E * 应变
        D = self._get_constitutive_matrix()
        stress = D * strain
        
        logger.debug(f"单元 {self.id} 应力: {stress:.2e} Pa")
        return stress
    
    def compute_strain(self, displacements: np.ndarray) -> float:
        """
        计算单元应变
        
        ε = B * u
        
        Args:
            displacements: 单元节点位移向量 [u_i, v_i, u_j, v_j]
            
        Returns:
            单元轴向应变
        """
        B = self._get_strain_displacement_matrix()
        strain = (B @ displacements)[0]
        return strain
    
    def compute_internal_force(self, displacements: np.ndarray) -> float:
        """
        计算单元内力（轴力）
        
        N = σ * A
        
        Args:
            displacements: 单元节点位移向量
            
        Returns:
            单元轴力 (正为拉力，负为压力)
        """
        stress = self.compute_stress(displacements)
        return stress * self.area
    
    def compute_element_forces(self, displacements: np.ndarray) -> np.ndarray:
        """
        计算单元节点力向量
        
        f = K * u
        
        Args:
            displacements: 单元节点位移向量
            
        Returns:
            4x1 节点力向量
        """
        ke = self.compute_stiffness_matrix()
        return ke @ displacements
    
    def __repr__(self) -> str:
        return (f"TrussElement(id={self.id}, nodes=({self.node_i},{self.node_j}), "
                f"L={self._length:.4f}m, A={self.area:.6f}m²)")
