"""CST 平面应力三角形单元类定义

理论基础：
CST（Constant Strain Triangle）是最基本的平面应力/应变三角形单元，
用于二维连续体的有限元分析。

基本假设：
1. 单元内应变为常数（线性位移场）
2. 平面应力状态：σz = τxz = τyz = 0
3. 材料为各向同性线弹性
4. 小变形假设

位移插值：
u(x,y) = N1*u1 + N2*u2 + N3*u3
v(x,y) = N1*v1 + N2*v2 + N3*v3

其中 Ni 为面积坐标形函数：
N1 = A1/A, N2 = A2/A, N3 = A3/A

应变-位移关系：
ε = B * u
其中 B 为 3x6 应变-位移矩阵

本构关系（平面应力）：
σ = D * ε
其中 D 为弹性矩阵

参考文献：
- Zienkiewicz, O.C., Taylor, R.L. "The Finite Element Method"
- Cook, R.D. "Concepts and Applications of Finite Element Analysis"
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, TYPE_CHECKING, List

if TYPE_CHECKING:
    from .material import Material

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class CSTElement:
    """
    CST 平面应力三角形单元类
    
    实现二维平面应力状态下的常应变三角形单元。
    每个节点有 2 个自由度（u, v），单元共 6 个自由度。
    
    Attributes:
        id: 单元编号
        node_ids: 三个节点编号 (i, j, k)，按逆时针排列
        material: 材料属性
        thickness: 单元厚度 (m)
        coords: 三个节点坐标 [(x1,y1), (x2,y2), (x3,y3)]
    """
    id: int
    node_ids: Tuple[int, int, int]
    material: 'Material'
    thickness: float
    coords: List[Tuple[float, float]] = field(default_factory=list)
    
    # 计算属性（延迟初始化）
    _area: float = field(init=False, repr=False, default=0.0)
    _B: np.ndarray = field(init=False, repr=False, default=None)
    _D: np.ndarray = field(init=False, repr=False, default=None)
    
    def __post_init__(self):
        """初始化几何属性"""
        if self.thickness <= 0:
            raise ValueError(f"单元厚度必须为正值，当前值: {self.thickness}")
        if len(self.coords) == 3:
            self._compute_geometry()
    
    def update_coordinates(self, coords: List[Tuple[float, float]]):
        """更新节点坐标并重新计算几何属性"""
        if len(coords) != 3:
            raise ValueError("CST 单元需要 3 个节点坐标")
        self.coords = coords
        self._compute_geometry()
    
    def _compute_geometry(self):
        """计算单元几何属性和矩阵"""
        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
        x3, y3 = self.coords[2]
        
        # 计算面积（使用行列式公式）
        # 2A = |x1 y1 1|
        #      |x2 y2 1|
        #      |x3 y3 1|
        self._area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        
        if self._area < 1e-12:
            raise ValueError(f"单元 {self.id} 面积过小或节点共线: {self._area}")
        
        # 计算应变-位移矩阵 B
        self._compute_B_matrix()
        
        # 计算弹性矩阵 D
        self._compute_D_matrix()
    
    def _compute_B_matrix(self):
        """
        计算应变-位移矩阵 B (3x6)
        
        ε = [εx, εy, γxy]^T = B * [u1, v1, u2, v2, u3, v3]^T
        
        B = (1/2A) * [b1  0  b2  0  b3  0 ]
                     [0   c1  0  c2  0  c3]
                     [c1  b1 c2  b2 c3  b3]
        
        其中：
        bi = yj - yk
        ci = xk - xj
        """
        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
        x3, y3 = self.coords[2]
        
        # 计算 b 和 c 系数
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1
        
        # 组装 B 矩阵
        A2 = 2 * self._area
        self._B = (1 / A2) * np.array([
            [b1,  0, b2,  0, b3,  0],
            [ 0, c1,  0, c2,  0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])
    
    def _compute_D_matrix(self):
        """
        计算弹性矩阵 D (3x3)
        
        平面应力状态：
        D = E/(1-ν²) * [1   ν   0      ]
                       [ν   1   0      ]
                       [0   0   (1-ν)/2]
        """
        E = self.material.E
        nu = self.material.nu
        
        coeff = E / (1 - nu**2)
        self._D = coeff * np.array([
            [1,  nu, 0],
            [nu, 1,  0],
            [0,  0,  (1 - nu) / 2]
        ])
    
    @property
    def area(self) -> float:
        """单元面积"""
        return self._area
    
    @property
    def volume(self) -> float:
        """单元体积"""
        return self._area * self.thickness
    
    def get_strain_displacement_matrix(self) -> np.ndarray:
        """获取应变-位移矩阵 B (3x6)"""
        return self._B.copy()
    
    def get_constitutive_matrix(self) -> np.ndarray:
        """获取弹性矩阵 D (3x3)"""
        return self._D.copy()
    
    def compute_stiffness_matrix(self) -> np.ndarray:
        """
        计算单元刚度矩阵（整体坐标系）
        
        K = ∫∫ B^T * D * B * t dA = B^T * D * B * t * A
        
        由于 CST 单元的 B 矩阵为常数，积分简化为乘以面积。
        
        Returns:
            6x6 单元刚度矩阵
        """
        # K = B^T * D * B * t * A
        ke = self._B.T @ self._D @ self._B * self.thickness * self._area
        
        logger.debug(f"CST 单元 {self.id} 刚度矩阵计算完成")
        return ke
    
    def compute_mass_matrix(self, consistent: bool = True) -> np.ndarray:
        """
        计算单元质量矩阵
        
        Args:
            consistent: True 使用一致质量矩阵，False 使用集中质量矩阵
            
        Returns:
            6x6 单元质量矩阵
        """
        m_total = self.material.rho * self._area * self.thickness
        
        if consistent:
            # 一致质量矩阵
            # M = ρ * t * A * ∫∫ N^T * N dA
            # 对于 CST 单元：
            # M = (ρ*t*A/12) * [2 0 1 0 1 0]
            #                  [0 2 0 1 0 1]
            #                  [1 0 2 0 1 0]
            #                  [0 1 0 2 0 1]
            #                  [1 0 1 0 2 0]
            #                  [0 1 0 1 0 2]
            me = (m_total / 12) * np.array([
                [2, 0, 1, 0, 1, 0],
                [0, 2, 0, 1, 0, 1],
                [1, 0, 2, 0, 1, 0],
                [0, 1, 0, 2, 0, 1],
                [1, 0, 1, 0, 2, 0],
                [0, 1, 0, 1, 0, 2]
            ], dtype=float)
        else:
            # 集中质量矩阵
            m_node = m_total / 3
            me = m_node * np.eye(6)
        
        logger.debug(f"CST 单元 {self.id} 质量矩阵计算完成, m_total={m_total:.4f}")
        return me
    
    def compute_stress(self, displacements: np.ndarray) -> np.ndarray:
        """
        计算单元应力
        
        σ = D * ε = D * B * u
        
        Args:
            displacements: 单元节点位移向量 [u1, v1, u2, v2, u3, v3]
            
        Returns:
            应力向量 [σx, σy, τxy] (Pa)
        """
        if len(displacements) != 6:
            raise ValueError(f"位移向量长度必须为6，当前: {len(displacements)}")
        
        # 计算应变
        strain = self._B @ displacements
        
        # 计算应力
        stress = self._D @ strain
        
        logger.debug(f"CST 单元 {self.id} 应力: σx={stress[0]:.2e}, σy={stress[1]:.2e}, τxy={stress[2]:.2e}")
        return stress
    
    def compute_strain(self, displacements: np.ndarray) -> np.ndarray:
        """
        计算单元应变
        
        ε = B * u
        
        Args:
            displacements: 单元节点位移向量 [u1, v1, u2, v2, u3, v3]
            
        Returns:
            应变向量 [εx, εy, γxy]
        """
        return self._B @ displacements
    
    def compute_von_mises_stress(self, displacements: np.ndarray) -> float:
        """
        计算 von Mises 等效应力
        
        σ_vm = √(σx² + σy² - σx*σy + 3*τxy²)
        
        Args:
            displacements: 单元节点位移向量
            
        Returns:
            von Mises 应力 (Pa)
        """
        stress = self.compute_stress(displacements)
        sigma_x, sigma_y, tau_xy = stress
        
        von_mises = np.sqrt(sigma_x**2 + sigma_y**2 - sigma_x * sigma_y + 3 * tau_xy**2)
        return von_mises
    
    def compute_principal_stresses(self, displacements: np.ndarray) -> Tuple[float, float, float]:
        """
        计算主应力和主方向
        
        Args:
            displacements: 单元节点位移向量
            
        Returns:
            (σ1, σ2, θ) - 主应力和主方向角（弧度）
        """
        stress = self.compute_stress(displacements)
        sigma_x, sigma_y, tau_xy = stress
        
        # 主应力
        sigma_avg = (sigma_x + sigma_y) / 2
        R = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
        
        sigma_1 = sigma_avg + R  # 最大主应力
        sigma_2 = sigma_avg - R  # 最小主应力
        
        # 主方向角
        if abs(sigma_x - sigma_y) < 1e-12:
            theta = np.pi / 4 if tau_xy > 0 else -np.pi / 4
        else:
            theta = 0.5 * np.arctan2(2 * tau_xy, sigma_x - sigma_y)
        
        return sigma_1, sigma_2, theta
    
    def compute_element_forces(self, displacements: np.ndarray) -> np.ndarray:
        """
        计算单元节点力向量
        
        f = K * u
        
        Args:
            displacements: 单元节点位移向量
            
        Returns:
            6x1 节点力向量
        """
        ke = self.compute_stiffness_matrix()
        return ke @ displacements
    
    def get_centroid(self) -> Tuple[float, float]:
        """获取单元形心坐标"""
        x_c = sum(c[0] for c in self.coords) / 3
        y_c = sum(c[1] for c in self.coords) / 3
        return x_c, y_c
    
    def __repr__(self) -> str:
        return (f"CSTElement(id={self.id}, nodes={self.node_ids}, "
                f"A={self._area:.6f}m², t={self.thickness:.4f}m)")
