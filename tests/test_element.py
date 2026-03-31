"""单元类测试"""
import pytest
import numpy as np
from fem_truss.core import TrussElement, Material


@pytest.fixture
def steel():
    """钢材材料"""
    return Material(id=1, name="Steel", E=2.06e11, nu=0.3, rho=7850)


@pytest.fixture
def horizontal_element(steel):
    """水平单元"""
    return TrussElement(
        id=1,
        node_i=1,
        node_j=2,
        material=steel,
        area=0.001,
        coords_i=(0.0, 0.0),
        coords_j=(1.0, 0.0)
    )


@pytest.fixture
def inclined_element(steel):
    """45度倾斜单元"""
    return TrussElement(
        id=2,
        node_i=1,
        node_j=2,
        material=steel,
        area=0.001,
        coords_i=(0.0, 0.0),
        coords_j=(1.0, 1.0)
    )


class TestTrussElement:
    """桁架单元测试类"""
    
    def test_element_creation(self, horizontal_element):
        """测试单元创建"""
        assert horizontal_element.id == 1
        assert horizontal_element.node_i == 1
        assert horizontal_element.node_j == 2
        assert horizontal_element.area == 0.001
    
    def test_element_length(self, horizontal_element, inclined_element):
        """测试单元长度计算"""
        assert abs(horizontal_element.length - 1.0) < 1e-10
        assert abs(inclined_element.length - np.sqrt(2)) < 1e-10
    
    def test_direction_cosines(self, horizontal_element, inclined_element):
        """测试方向余弦"""
        # 水平单元
        assert abs(horizontal_element.cos_theta - 1.0) < 1e-10
        assert abs(horizontal_element.sin_theta - 0.0) < 1e-10
        
        # 45度单元
        assert abs(inclined_element.cos_theta - np.sqrt(2)/2) < 1e-10
        assert abs(inclined_element.sin_theta - np.sqrt(2)/2) < 1e-10
    
    def test_stiffness_matrix_shape(self, horizontal_element):
        """测试刚度矩阵形状"""
        ke = horizontal_element.compute_stiffness_matrix()
        assert ke.shape == (4, 4)
    
    def test_stiffness_matrix_symmetry(self, horizontal_element):
        """测试刚度矩阵对称性"""
        ke = horizontal_element.compute_stiffness_matrix()
        assert np.allclose(ke, ke.T)
    
    def test_stiffness_matrix_horizontal(self, horizontal_element):
        """测试水平单元刚度矩阵"""
        ke = horizontal_element.compute_stiffness_matrix()
        k = horizontal_element.material.E * horizontal_element.area / horizontal_element.length
        
        # 水平单元的刚度矩阵应该只有 x 方向分量
        expected = k * np.array([
            [ 1,  0, -1,  0],
            [ 0,  0,  0,  0],
            [-1,  0,  1,  0],
            [ 0,  0,  0,  0]
        ])
        assert np.allclose(ke, expected)
    
    def test_mass_matrix_shape(self, horizontal_element):
        """测试质量矩阵形状"""
        me = horizontal_element.compute_mass_matrix()
        assert me.shape == (4, 4)
    
    def test_mass_matrix_symmetry(self, horizontal_element):
        """测试质量矩阵对称性"""
        me = horizontal_element.compute_mass_matrix()
        assert np.allclose(me, me.T)
    
    def test_stress_calculation(self, horizontal_element):
        """测试应力计算"""
        # 单元拉伸 0.001m
        displacements = np.array([0.0, 0.0, 0.001, 0.0])
        stress = horizontal_element.compute_stress(displacements)
        
        # 应力 = E * 应变 = E * (delta_L / L)
        expected_stress = horizontal_element.material.E * 0.001 / 1.0
        assert abs(stress - expected_stress) < 1e-6
    
    def test_strain_calculation(self, horizontal_element):
        """测试应变计算"""
        displacements = np.array([0.0, 0.0, 0.001, 0.0])
        strain = horizontal_element.compute_strain(displacements)
        
        expected_strain = 0.001 / 1.0
        assert abs(strain - expected_strain) < 1e-10
    
    def test_invalid_area(self, steel):
        """测试无效截面面积"""
        with pytest.raises(ValueError):
            TrussElement(
                id=1, node_i=1, node_j=2, material=steel, area=-0.001,
                coords_i=(0, 0), coords_j=(1, 0)
            )
    
    def test_zero_length_element(self, steel):
        """测试零长度单元"""
        with pytest.raises(ValueError):
            TrussElement(
                id=1, node_i=1, node_j=2, material=steel, area=0.001,
                coords_i=(0, 0), coords_j=(0, 0)
            )
