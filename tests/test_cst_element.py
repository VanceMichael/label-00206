"""CST 平面应力三角形单元测试"""
import pytest
import numpy as np
from fem_truss.core import CSTElement, Material


class TestCSTElement:
    """CST 单元测试类"""
    
    @pytest.fixture
    def steel(self):
        """钢材材料"""
        return Material(id=1, name="Steel", E=2.0e11, nu=0.3, rho=7850)
    
    @pytest.fixture
    def equilateral_triangle(self, steel):
        """等边三角形单元"""
        coords = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.866)]
        return CSTElement(
            id=1,
            node_ids=(1, 2, 3),
            material=steel,
            thickness=0.01,
            coords=coords
        )
    
    @pytest.fixture
    def right_triangle(self, steel):
        """直角三角形单元"""
        coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        return CSTElement(
            id=2,
            node_ids=(1, 2, 3),
            material=steel,
            thickness=0.02,
            coords=coords
        )
    
    def test_element_creation(self, equilateral_triangle):
        """测试单元创建"""
        elem = equilateral_triangle
        assert elem.id == 1
        assert elem.node_ids == (1, 2, 3)
        assert elem.thickness == 0.01
    
    def test_element_area(self, equilateral_triangle, right_triangle):
        """测试单元面积计算"""
        # 等边三角形面积 ≈ 0.433
        assert abs(equilateral_triangle.area - 0.433) < 0.001
        
        # 直角三角形面积 = 0.5
        assert abs(right_triangle.area - 0.5) < 1e-10
    
    def test_element_volume(self, right_triangle):
        """测试单元体积计算"""
        # 体积 = 面积 × 厚度 = 0.5 × 0.02 = 0.01
        assert abs(right_triangle.volume - 0.01) < 1e-10
    
    def test_stiffness_matrix_shape(self, equilateral_triangle):
        """测试刚度矩阵形状"""
        ke = equilateral_triangle.compute_stiffness_matrix()
        assert ke.shape == (6, 6)
    
    def test_stiffness_matrix_symmetry(self, equilateral_triangle):
        """测试刚度矩阵对称性"""
        ke = equilateral_triangle.compute_stiffness_matrix()
        # 允许小的数值误差
        assert np.allclose(ke, ke.T, rtol=1e-10)
    
    def test_stiffness_matrix_positive_definite(self, equilateral_triangle):
        """测试刚度矩阵正定性（特征值应为非负）"""
        ke = equilateral_triangle.compute_stiffness_matrix()
        eigenvalues = np.linalg.eigvalsh(ke)
        # 由于刚体模式，最小特征值可能接近零
        assert all(eigenvalues >= -1e-10)
    
    def test_mass_matrix_shape(self, equilateral_triangle):
        """测试质量矩阵形状"""
        me = equilateral_triangle.compute_mass_matrix()
        assert me.shape == (6, 6)
    
    def test_mass_matrix_symmetry(self, equilateral_triangle):
        """测试质量矩阵对称性"""
        me = equilateral_triangle.compute_mass_matrix()
        assert np.allclose(me, me.T, rtol=1e-10)
    
    def test_mass_matrix_consistent_vs_lumped(self, equilateral_triangle):
        """测试一致质量矩阵与集中质量矩阵"""
        me_consistent = equilateral_triangle.compute_mass_matrix(consistent=True)
        me_lumped = equilateral_triangle.compute_mass_matrix(consistent=False)
        
        # 总质量应相同
        total_mass_consistent = np.trace(me_consistent) / 2  # 每个节点2个自由度
        total_mass_lumped = np.trace(me_lumped) / 2
        
        # 集中质量矩阵是对角矩阵
        assert np.allclose(me_lumped, np.diag(np.diag(me_lumped)))
    
    def test_strain_displacement_matrix(self, right_triangle):
        """测试应变-位移矩阵"""
        B = right_triangle.get_strain_displacement_matrix()
        assert B.shape == (3, 6)
    
    def test_constitutive_matrix(self, right_triangle):
        """测试弹性矩阵"""
        D = right_triangle.get_constitutive_matrix()
        assert D.shape == (3, 3)
        # 弹性矩阵应对称
        assert np.allclose(D, D.T)
    
    def test_stress_calculation(self, right_triangle):
        """测试应力计算"""
        # 施加单位位移
        displacements = np.array([0.0, 0.0, 1e-4, 0.0, 0.0, 0.0])
        stress = right_triangle.compute_stress(displacements)
        
        assert len(stress) == 3  # σx, σy, τxy
        assert stress[0] != 0  # 应有 x 方向应力
    
    def test_strain_calculation(self, right_triangle):
        """测试应变计算"""
        displacements = np.array([0.0, 0.0, 1e-4, 0.0, 0.0, 0.0])
        strain = right_triangle.compute_strain(displacements)
        
        assert len(strain) == 3  # εx, εy, γxy
    
    def test_von_mises_stress(self, right_triangle):
        """测试 von Mises 应力计算"""
        displacements = np.array([0.0, 0.0, 1e-4, 0.0, 0.0, 1e-4])
        von_mises = right_triangle.compute_von_mises_stress(displacements)
        
        assert von_mises >= 0  # von Mises 应力应为非负
    
    def test_principal_stresses(self, right_triangle):
        """测试主应力计算"""
        displacements = np.array([0.0, 0.0, 1e-4, 0.0, 0.0, 0.0])
        sigma_1, sigma_2, theta = right_triangle.compute_principal_stresses(displacements)
        
        # σ1 >= σ2
        assert sigma_1 >= sigma_2
        # 主方向角在 [-π/2, π/2] 范围内
        assert -np.pi/2 <= theta <= np.pi/2
    
    def test_centroid(self, right_triangle):
        """测试形心计算"""
        x_c, y_c = right_triangle.get_centroid()
        # 直角三角形形心 = (1/3, 1/3)
        assert abs(x_c - 1/3) < 1e-10
        assert abs(y_c - 1/3) < 1e-10
    
    def test_invalid_thickness(self, steel):
        """测试无效厚度"""
        with pytest.raises(ValueError):
            CSTElement(
                id=1,
                node_ids=(1, 2, 3),
                material=steel,
                thickness=-0.01,
                coords=[(0, 0), (1, 0), (0.5, 0.866)]
            )
    
    def test_collinear_nodes(self, steel):
        """测试共线节点（面积为零）"""
        with pytest.raises(ValueError):
            CSTElement(
                id=1,
                node_ids=(1, 2, 3),
                material=steel,
                thickness=0.01,
                coords=[(0, 0), (1, 0), (2, 0)]  # 三点共线
            )
