"""求解器测试"""
import pytest
import numpy as np
from fem_truss.core import TrussStructure, Material
from fem_truss.solver import StaticSolver, DynamicSolver


@pytest.fixture
def simple_truss():
    """简单三角形桁架"""
    structure = TrussStructure()
    
    steel = Material(id=1, name="Steel", E=2.06e11, nu=0.3, rho=7850)
    structure.add_material(steel)
    
    structure.add_node(1, 0.0, 0.0)
    structure.add_node(2, 4.0, 0.0)
    structure.add_node(3, 2.0, 2.0)
    
    structure.add_element(1, 1, 2, material_id=1, area=0.001)
    structure.add_element(2, 1, 3, material_id=1, area=0.001)
    structure.add_element(3, 2, 3, material_id=1, area=0.001)
    
    structure.apply_boundary(1, fix_x=True, fix_y=True)
    structure.apply_boundary(2, fix_x=False, fix_y=True)
    
    structure.apply_load(3, fx=0, fy=-10000)
    
    return structure


@pytest.fixture
def cantilever_truss():
    """悬臂桁架（用于验证）"""
    structure = TrussStructure()
    
    # E = 200 GPa, A = 0.001 m²
    steel = Material(id=1, name="Steel", E=2.0e11, nu=0.3, rho=7850)
    structure.add_material(steel)
    
    # 简单两节点悬臂
    structure.add_node(1, 0.0, 0.0)
    structure.add_node(2, 1.0, 0.0)
    
    structure.add_element(1, 1, 2, material_id=1, area=0.001)
    
    structure.apply_boundary(1, fix_x=True, fix_y=True)
    structure.apply_load(2, fx=1000, fy=0)  # 1kN 水平力
    
    return structure


class TestStaticSolver:
    """静力求解器测试"""
    
    def test_solver_creation(self, simple_truss):
        """测试求解器创建"""
        solver = StaticSolver(simple_truss)
        assert solver.structure == simple_truss
    
    def test_static_solve(self, simple_truss):
        """测试静力求解"""
        solver = StaticSolver(simple_truss)
        result = solver.solve()
        
        assert result.displacements is not None
        assert len(result.displacements) == 6
        assert len(result.stresses) == 3
    
    def test_cantilever_displacement(self, cantilever_truss):
        """测试悬臂位移（理论验证）"""
        solver = StaticSolver(cantilever_truss)
        result = solver.solve()
        
        # 理论位移: u = PL / (EA) = 1000 * 1 / (2e11 * 0.001) = 5e-6 m
        expected_u = 1000 * 1.0 / (2.0e11 * 0.001)
        
        # 节点2的x位移 (dof 2)
        actual_u = result.displacements[2]
        
        assert abs(actual_u - expected_u) / expected_u < 0.01  # 1% 误差
    
    def test_cantilever_stress(self, cantilever_truss):
        """测试悬臂应力（理论验证）"""
        solver = StaticSolver(cantilever_truss)
        result = solver.solve()
        
        # 理论应力: σ = P / A = 1000 / 0.001 = 1e6 Pa = 1 MPa
        expected_stress = 1000 / 0.001
        
        actual_stress = result.stresses[1]
        
        assert abs(actual_stress - expected_stress) / expected_stress < 0.01
    
    def test_equilibrium(self, simple_truss):
        """测试力平衡"""
        solver = StaticSolver(simple_truss)
        result = solver.solve()
        
        # 反力之和应等于外力
        reactions = result.reactions
        constrained_dofs = simple_truss.get_constrained_dofs()
        
        total_reaction_y = sum(reactions[dof] for dof in constrained_dofs if dof % 2 == 1)
        applied_force_y = -10000
        
        # 反力应平衡外力
        assert abs(total_reaction_y + applied_force_y) < 1e-6
    
    def test_strain_energy(self, simple_truss):
        """测试应变能"""
        solver = StaticSolver(simple_truss)
        result = solver.solve()
        
        energy = solver.compute_strain_energy(result.displacements)
        
        # 应变能应为正值
        assert energy > 0
    
    def test_modal_analysis(self, simple_truss):
        """测试模态分析"""
        solver = StaticSolver(simple_truss)
        frequencies, mode_shapes = solver.modal_analysis(num_modes=3)
        
        assert len(frequencies) == 3
        assert all(f >= 0 for f in frequencies)  # 频率非负
        assert frequencies[0] <= frequencies[1] <= frequencies[2]  # 频率递增


class TestDynamicSolver:
    """动力求解器测试"""
    
    def test_solver_creation(self, simple_truss):
        """测试求解器创建"""
        solver = DynamicSolver(simple_truss)
        assert solver.structure == simple_truss
    
    def test_newmark_params(self, simple_truss):
        """测试 Newmark 参数设置"""
        solver = DynamicSolver(simple_truss)
        solver.set_newmark_params(gamma=0.5, beta=0.25)
        
        assert solver.gamma == 0.5
        assert solver.beta == 0.25
    
    def test_rayleigh_damping(self, simple_truss):
        """测试 Rayleigh 阻尼设置"""
        solver = DynamicSolver(simple_truss)
        solver.set_rayleigh_damping(xi1=0.05, xi2=0.05, omega1=10, omega2=100)
        
        assert solver.alpha > 0
        assert solver.beta_damping > 0
    
    def test_dynamic_solve(self, simple_truss):
        """测试动力求解"""
        solver = DynamicSolver(simple_truss)
        solver.set_damping_ratio(0.05)
        
        # 简谐激励
        time = np.linspace(0, 0.5, 50)
        acceleration = 0.1 * 9.81 * np.sin(2 * np.pi * 2 * time)
        
        result = solver.solve(acceleration, time, direction='x')
        
        assert result.time is not None
        assert result.displacement_history.shape == (50, 6)
        assert result.velocity_history.shape == (50, 6)
        assert result.acceleration_history.shape == (50, 6)
    
    def test_initial_conditions(self, simple_truss):
        """测试初始条件"""
        solver = DynamicSolver(simple_truss)
        solver.set_damping_ratio(0.05)
        
        time = np.linspace(0, 0.1, 10)
        acceleration = np.zeros_like(time)
        
        result = solver.solve(acceleration, time, direction='x')
        
        # 无激励时，位移应保持为零
        assert np.max(np.abs(result.displacement_history)) < 1e-10
    
    def test_damping_effect(self, simple_truss):
        """测试阻尼效果"""
        solver = DynamicSolver(simple_truss)
        
        time = np.linspace(0, 2, 200)
        # 脉冲激励
        acceleration = np.zeros_like(time)
        acceleration[0:10] = 1.0
        
        # 无阻尼
        solver.alpha = 0
        solver.beta_damping = 0
        result_undamped = solver.solve(acceleration, time, direction='x')
        
        # 有阻尼
        solver.set_damping_ratio(0.1)
        result_damped = solver.solve(acceleration, time, direction='x')
        
        # 有阻尼的响应应该衰减
        max_undamped = np.max(np.abs(result_undamped.displacement_history[-50:]))
        max_damped = np.max(np.abs(result_damped.displacement_history[-50:]))
        
        assert max_damped < max_undamped
