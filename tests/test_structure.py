"""结构类测试"""
import pytest
import numpy as np
from fem_truss.core import TrussStructure, Material


@pytest.fixture
def simple_truss():
    """简单三角形桁架"""
    structure = TrussStructure()
    
    # 添加材料
    steel = Material(id=1, name="Steel", E=2.06e11, nu=0.3, rho=7850)
    structure.add_material(steel)
    
    # 添加节点
    structure.add_node(1, 0.0, 0.0)
    structure.add_node(2, 4.0, 0.0)
    structure.add_node(3, 2.0, 2.0)
    
    # 添加单元
    structure.add_element(1, 1, 2, material_id=1, area=0.001)
    structure.add_element(2, 1, 3, material_id=1, area=0.001)
    structure.add_element(3, 2, 3, material_id=1, area=0.001)
    
    # 边界条件
    structure.apply_boundary(1, fix_x=True, fix_y=True)
    structure.apply_boundary(2, fix_x=False, fix_y=True)
    
    # 荷载
    structure.apply_load(3, fx=0, fy=-10000)
    
    return structure


class TestTrussStructure:
    """桁架结构测试类"""
    
    def test_structure_creation(self):
        """测试结构创建"""
        structure = TrussStructure()
        assert structure.num_nodes == 0
        assert structure.num_elements == 0
    
    def test_add_node(self):
        """测试添加节点"""
        structure = TrussStructure()
        node = structure.add_node(1, 1.0, 2.0)
        
        assert structure.num_nodes == 1
        assert node.x == 1.0
        assert node.y == 2.0
    
    def test_add_duplicate_node(self):
        """测试添加重复节点"""
        structure = TrussStructure()
        structure.add_node(1, 0, 0)
        
        with pytest.raises(Exception):
            structure.add_node(1, 1, 1)
    
    def test_add_element(self, simple_truss):
        """测试添加单元"""
        assert simple_truss.num_elements == 3
    
    def test_ndof(self, simple_truss):
        """测试自由度数"""
        assert simple_truss.ndof == 6  # 3 nodes * 2 dof
    
    def test_dof_mapping(self, simple_truss):
        """测试自由度映射"""
        dof_1 = simple_truss.get_dof_indices(1)
        dof_2 = simple_truss.get_dof_indices(2)
        dof_3 = simple_truss.get_dof_indices(3)
        
        assert dof_1 == (0, 1)
        assert dof_2 == (2, 3)
        assert dof_3 == (4, 5)
    
    def test_stiffness_matrix_assembly(self, simple_truss):
        """测试刚度矩阵组装"""
        K = simple_truss.assemble_stiffness_matrix()
        
        assert K.shape == (6, 6)
        assert np.allclose(K, K.T)  # 对称性
    
    def test_mass_matrix_assembly(self, simple_truss):
        """测试质量矩阵组装"""
        M = simple_truss.assemble_mass_matrix()
        
        assert M.shape == (6, 6)
        assert np.allclose(M, M.T)  # 对称性
    
    def test_load_vector_assembly(self, simple_truss):
        """测试荷载向量组装"""
        F = simple_truss.assemble_load_vector()
        
        assert F.shape == (6,)
        assert F[5] == -10000  # 节点3的y方向荷载
    
    def test_constrained_dofs(self, simple_truss):
        """测试约束自由度"""
        constrained = simple_truss.get_constrained_dofs()
        
        # 节点1: fix_x, fix_y -> dof 0, 1
        # 节点2: fix_y -> dof 3
        assert 0 in constrained
        assert 1 in constrained
        assert 3 in constrained
        assert len(constrained) == 3
    
    def test_free_dofs(self, simple_truss):
        """测试自由自由度"""
        free = simple_truss.get_free_dofs()
        
        # 自由度: 2 (节点2的x), 4 (节点3的x), 5 (节点3的y)
        assert 2 in free
        assert 4 in free
        assert 5 in free
        assert len(free) == 3
    
    def test_validation(self, simple_truss):
        """测试结构验证"""
        assert simple_truss.validate() == True
    
    def test_validation_no_boundary(self):
        """测试无边界条件验证"""
        structure = TrussStructure()
        structure.add_material(Material(id=1, name="Steel", E=2.06e11))
        structure.add_node(1, 0, 0)
        structure.add_node(2, 1, 0)
        structure.add_element(1, 1, 2, material_id=1, area=0.001)
        structure.apply_load(2, fy=-1000)
        
        assert structure.validate() == False
