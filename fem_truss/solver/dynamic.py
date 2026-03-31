"""动力求解器"""
import numpy as np
from scipy.linalg import solve, eigh
from typing import Optional, Tuple

from ..core import TrussStructure
from ..io.writer import DynamicResult
from ..utils import get_logger, SolverError

logger = get_logger(__name__)


class DynamicSolver:
    """
    动力求解器
    
    基于 Newmark-β 法求解桁架结构的动力响应
    """
    
    def __init__(self, structure: TrussStructure):
        """
        初始化求解器
        
        Args:
            structure: 桁架结构对象
        """
        self.structure = structure
        
        # Newmark-β 参数（平均加速度法）
        self.gamma = 0.5
        self.beta = 0.25
        
        # Rayleigh 阻尼参数
        self.alpha = 0.0  # 质量阻尼系数
        self.beta_damping = 0.0  # 刚度阻尼系数
        
    def set_newmark_params(self, gamma: float = 0.5, beta: float = 0.25):
        """
        设置 Newmark-β 参数
        
        Args:
            gamma: γ 参数 (默认 0.5)
            beta: β 参数 (默认 0.25，平均加速度法)
        """
        self.gamma = gamma
        self.beta = beta
        logger.info(f"Newmark 参数: γ={gamma}, β={beta}")
    
    def set_rayleigh_damping(self, xi1: float, xi2: float, 
                             omega1: float, omega2: float):
        """
        设置 Rayleigh 阻尼参数
        
        Args:
            xi1: 第一阶模态阻尼比
            xi2: 第二阶模态阻尼比
            omega1: 第一阶圆频率 (rad/s)
            omega2: 第二阶圆频率 (rad/s)
        """
        # 求解 Rayleigh 阻尼系数
        # C = alpha * M + beta * K
        # xi_i = alpha / (2 * omega_i) + beta * omega_i / 2
        
        A = np.array([
            [1 / (2 * omega1), omega1 / 2],
            [1 / (2 * omega2), omega2 / 2]
        ])
        b = np.array([xi1, xi2])
        
        coeffs = solve(A, b)
        self.alpha = coeffs[0]
        self.beta_damping = coeffs[1]
        
        logger.info(f"Rayleigh 阻尼系数: α={self.alpha:.4e}, β={self.beta_damping:.4e}")
    
    def set_damping_ratio(self, xi: float = 0.05):
        """
        基于目标阻尼比自动设置 Rayleigh 阻尼
        
        Args:
            xi: 目标阻尼比 (默认 5%)
        """
        # 先进行模态分析获取前两阶频率
        K = self.structure.assemble_stiffness_matrix()
        M = self.structure.assemble_mass_matrix()
        
        free_dofs = self.structure.get_free_dofs()
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        M_ff = M[np.ix_(free_dofs, free_dofs)]
        
        eigenvalues, _ = eigh(K_ff, M_ff)
        omega = np.sqrt(np.abs(eigenvalues))
        
        if len(omega) >= 2:
            omega1, omega2 = omega[0], omega[1]
        else:
            omega1 = omega[0]
            omega2 = omega1 * 2
        
        self.set_rayleigh_damping(xi, xi, omega1, omega2)
    
    def build_damping_matrix(self, M: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        构建阻尼矩阵
        
        Args:
            M: 质量矩阵
            K: 刚度矩阵
            
        Returns:
            阻尼矩阵
        """
        C = self.alpha * M + self.beta_damping * K
        logger.debug(f"阻尼矩阵构建完成")
        return C
    
    def solve(self, acceleration: np.ndarray, time: np.ndarray,
              direction: str = 'x',
              initial_displacement: Optional[np.ndarray] = None,
              initial_velocity: Optional[np.ndarray] = None) -> DynamicResult:
        """
        求解地震动响应（Newmark-β 法）
        
        Args:
            acceleration: 地震加速度时程 (m/s²)
            time: 时间序列 (s)
            direction: 地震作用方向 ('x' 或 'y')
            initial_displacement: 初始位移（可选）
            initial_velocity: 初始速度（可选）
            
        Returns:
            DynamicResult 包含位移、速度、加速度时程
        """
        logger.info("开始动力时程分析...")
        
        n_steps = len(time)
        dt = time[1] - time[0]
        ndof = self.structure.ndof
        
        logger.info(f"时间步数: {n_steps}, 时间步长: {dt:.4f}s, 总时长: {time[-1]:.2f}s")
        
        # 组装矩阵
        K = self.structure.assemble_stiffness_matrix()
        M = self.structure.assemble_mass_matrix()
        C = self.build_damping_matrix(M, K)
        
        # 获取自由度信息
        free_dofs = self.structure.get_free_dofs()
        constrained_dofs = self.structure.get_constrained_dofs()
        n_free = len(free_dofs)
        
        # 提取自由自由度子矩阵
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        M_ff = M[np.ix_(free_dofs, free_dofs)]
        C_ff = C[np.ix_(free_dofs, free_dofs)]
        
        # 构建地震荷载向量（惯性力）
        # F = -M * I * a_g
        influence_vector = np.zeros(n_free)
        for i, dof in enumerate(free_dofs):
            if direction == 'x' and dof % 2 == 0:
                influence_vector[i] = 1.0
            elif direction == 'y' and dof % 2 == 1:
                influence_vector[i] = 1.0
        
        # 初始化
        u = np.zeros(n_free) if initial_displacement is None else initial_displacement[free_dofs]
        v = np.zeros(n_free) if initial_velocity is None else initial_velocity[free_dofs]
        
        # 计算初始加速度
        F0 = -M_ff @ influence_vector * acceleration[0]
        a = solve(M_ff, F0 - C_ff @ v - K_ff @ u)
        
        # 存储结果
        displacement_history = np.zeros((n_steps, ndof))
        velocity_history = np.zeros((n_steps, ndof))
        acceleration_history = np.zeros((n_steps, ndof))
        
        displacement_history[0, free_dofs] = u
        velocity_history[0, free_dofs] = v
        acceleration_history[0, free_dofs] = a
        
        # Newmark-β 积分系数
        a0 = 1 / (self.beta * dt**2)
        a1 = self.gamma / (self.beta * dt)
        a2 = 1 / (self.beta * dt)
        a3 = 1 / (2 * self.beta) - 1
        a4 = self.gamma / self.beta - 1
        a5 = dt * (self.gamma / (2 * self.beta) - 1)
        a6 = dt * (1 - self.gamma)
        a7 = self.gamma * dt
        
        # 有效刚度矩阵
        K_eff = K_ff + a0 * M_ff + a1 * C_ff
        
        # 时间积分
        logger.info("开始时间积分...")
        for i in range(1, n_steps):
            # 地震荷载
            F_eq = -M_ff @ influence_vector * acceleration[i]
            
            # 有效荷载
            F_eff = F_eq + M_ff @ (a0 * u + a2 * v + a3 * a) + \
                    C_ff @ (a1 * u + a4 * v + a5 * a)
            
            # 求解位移增量
            u_new = solve(K_eff, F_eff)
            
            # 更新速度和加速度
            a_new = a0 * (u_new - u) - a2 * v - a3 * a
            v_new = v + a6 * a + a7 * a_new
            
            # 更新状态
            u, v, a = u_new, v_new, a_new
            
            # 存储结果
            displacement_history[i, free_dofs] = u
            velocity_history[i, free_dofs] = v
            acceleration_history[i, free_dofs] = a
            
            # 进度显示
            if i % (n_steps // 10) == 0:
                progress = i / n_steps * 100
                logger.info(f"积分进度: {progress:.0f}%")
        
        max_disp = np.max(np.abs(displacement_history))
        logger.info(f"动力分析完成，最大位移: {max_disp:.6e} m")
        
        return DynamicResult(
            time=time,
            displacement_history=displacement_history,
            velocity_history=velocity_history,
            acceleration_history=acceleration_history
        )
    
    def solve_harmonic(self, amplitude: float, frequency: float,
                       duration: float, dt: float = 0.01,
                       node_id: int = None, direction: str = 'y') -> DynamicResult:
        """
        求解简谐荷载响应
        
        Args:
            amplitude: 荷载幅值 (N)
            frequency: 荷载频率 (Hz)
            duration: 分析时长 (s)
            dt: 时间步长 (s)
            node_id: 荷载作用节点
            direction: 荷载方向
            
        Returns:
            DynamicResult
        """
        time = np.arange(0, duration, dt)
        omega = 2 * np.pi * frequency
        
        # 生成简谐荷载时程
        load_history = amplitude * np.sin(omega * time)
        
        logger.info(f"简谐荷载分析: 幅值={amplitude}N, 频率={frequency}Hz")
        
        # 转换为等效加速度（简化处理）
        # 这里假设荷载直接作用，不是地震输入
        # 实际应修改求解过程以支持节点荷载时程
        
        return self.solve(load_history / 1000, time, direction)
