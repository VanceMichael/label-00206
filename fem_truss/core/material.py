"""材料类定义"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Material:
    """
    材料属性类
    
    Attributes:
        id: 材料编号
        name: 材料名称
        E: 弹性模量 (Pa)
        nu: 泊松比
        rho: 密度 (kg/m³)
    """
    id: int
    name: str
    E: float  # 弹性模量
    nu: float = 0.3  # 泊松比
    rho: float = 7850.0  # 密度，默认钢材
    
    def __post_init__(self):
        """验证材料参数"""
        if self.E <= 0:
            raise ValueError(f"弹性模量必须为正值，当前值: {self.E}")
        if not 0 <= self.nu < 0.5:
            raise ValueError(f"泊松比必须在 [0, 0.5) 范围内，当前值: {self.nu}")
        if self.rho <= 0:
            raise ValueError(f"密度必须为正值，当前值: {self.rho}")
    
    @property
    def G(self) -> float:
        """剪切模量"""
        return self.E / (2 * (1 + self.nu))
    
    def __repr__(self) -> str:
        return f"Material(id={self.id}, name='{self.name}', E={self.E:.2e}, ρ={self.rho})"


# 预定义常用材料
STEEL_Q235 = Material(id=1, name="Q235钢", E=2.06e11, nu=0.3, rho=7850)
STEEL_Q345 = Material(id=2, name="Q345钢", E=2.06e11, nu=0.3, rho=7850)
ALUMINUM_6061 = Material(id=3, name="6061铝合金", E=6.9e10, nu=0.33, rho=2700)
