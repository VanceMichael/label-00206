"""静力求解器

支持密集矩阵和稀疏矩阵两种模式：
- 密集矩阵：适用于小规模问题（节点数 < 500）
- 稀疏矩阵：适用于中大规模问题（节点数 >= 500）

求解器选项：
- 直接求解器：scipy.linalg.solve（密集）或 scipy.sparse.linalg.spsolve（稀疏）
- 迭代求解器：共轭梯度法（CG），适用于大规模对称正定系统
"""
import numpy as np
from scipy.linalg import solve
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve, cg
from typing import Dict, Tuple, Optional

from ..core import TrussStructure
from ..io.writer import StaticResult
from ..utils import get_logger, SolverError

logger = get_logger(__name__)


class StaticSolver:
    """
    静力求解器
    
    基于有限元方法求解桁架结构的静力响应。
    支持密集矩阵和稀疏矩阵两种模式，以及直接求解和迭代求解两种方法。
    """
    
    def __init__(self, structure: TrussStructure, use_iterative: bool = False):
        """
        初始化求解器
        
        Args:
            structure: 桁架结构对象
            use_iterative: 是否使用迭代求解器（推荐大规模问题使用）
        """
        self.structure = structure
        self._K: np.ndarray = None  # 整体刚度矩阵
        self._F: np.ndarray = None  # 荷载向量
        self._use_iterative = use_iterative
        self._cg_tol = 1e-10  # 迭代求解器容差
        self._cg_maxiter = 1000  # 最大迭代次数
    
    def set_iterative_params(self, tol: float = 1e-10, maxiter: int = 1000):
        """
        设置迭代求解器参数
        
        Args:
            tol: 收敛容差
            maxiter: 最大迭代次数
        """
        self._cg_tol = tol
        self._cg_maxiter = maxiter
        logger.info(f"迭代求解器参数: tol={tol}, maxiter={maxiter}")
        
    def solve(self) -> StaticResult:
        """
        执行静力分析
        
        Returns:
            StaticResult 包含位移、应力、应变、反力
        """
        logger.info("开始静力分析...")
        
        # 验证结构
        if not self.structure.validate():
            logger.warning("结构验证未完全通过，继续求解...")
        
        # 组装矩阵
        self._K = self.structure.assemble_stiffness_matrix()
        self._F = self.structure.assemble_load_vector()
        
        # 获取自由度信息
        free_dofs = self.structure.get_free_dofs()
        constrained_dofs = self.structure.get_constrained_dofs()
        
        if len(free_dofs) == 0:
            raise SolverError("所有自由度都被约束，无法求解")
        
        logger.info(f"自由度: {len(free_dofs)}, 约束自由度: {len(constrained_dofs)}")
        
        # 提取自由自由度的子矩阵
        if issparse(self._K):
            K_ff = self._K[np.ix_(free_dofs, free_dofs)].tocsr()
        else:
            K_ff = self._K[np.ix_(free_dofs, free_dofs)]
        F_f = self._F[free_dofs]
        
        # 检查刚度矩阵条件（仅对密集矩阵）
        if not issparse(K_ff):
            cond_num = np.linalg.cond(K_ff)
            logger.info(f"刚度矩阵条件数: {cond_num:.2e}")
            
            if cond_num > 1e15:
                raise SolverError(f"刚度矩阵病态，条件数过大: {cond_num:.2e}")
        
        # 求解位移
        try:
            if issparse(K_ff):
                if self._use_iterative:
                    # 使用共轭梯度法（适用于大规模对称正定系统）
                    u_f, info = cg(K_ff, F_f, tol=self._cg_tol, maxiter=self._cg_maxiter)
                    if info > 0:
                        logger.warning(f"CG 求解器未收敛，迭代次数: {info}")
                    elif info < 0:
                        raise SolverError(f"CG 求解器出错: {info}")
                    logger.info(f"使用 CG 迭代求解器完成")
                else:
                    # 使用稀疏直接求解器
                    u_f = spsolve(K_ff, F_f)
                    logger.info(f"使用稀疏直接求解器完成")
            else:
                u_f = solve(K_ff, F_f)
                logger.info(f"使用密集直接求解器完成")
        except Exception as e:
            raise SolverError(f"线性方程组求解失败: {e}")
        
        # 组装完整位移向量
        displacements = np.zeros(self.structure.ndof)
        displacements[free_dofs] = u_f
        
        logger.info(f"最大位移: {np.max(np.abs(displacements)):.6e} m")
        
        # 计算支座反力
        if issparse(self._K):
            reactions = self._K @ displacements - self._F
        else:
            reactions = self._K @ displacements - self._F
        
        # 计算单元应力和应变
        stresses, strains = self._compute_element_results(displacements)
        
        logger.info("静力分析完成")
        
        return StaticResult(
            displacements=displacements,
            stresses=stresses,
            strains=strains,
            reactions=reactions
        )
    
    def _compute_element_results(self, displacements: np.ndarray) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        计算单元应力和应变
        
        Args:
            displacements: 节点位移向量
            
        Returns:
            (应力字典, 应变字典)
        """
        stresses = {}
        strains = {}
        
        for elem_id, elem in self.structure.elements.items():
            # 获取单元节点位移
            dof_indices = self.structure.get_element_dof_indices(elem)
            elem_disp = displacements[dof_indices]
            
            # 计算应力和应变
            stresses[elem_id] = elem.compute_stress(elem_disp)
            strains[elem_id] = elem.compute_strain(elem_disp)
        
        max_stress = max(abs(s) for s in stresses.values())
        logger.info(f"最大应力: {max_stress:.2e} Pa ({max_stress/1e6:.2f} MPa)")
        
        return stresses, strains
    
    def compute_strain_energy(self, displacements: np.ndarray) -> float:
        """
        计算应变能
        
        Args:
            displacements: 节点位移向量
            
        Returns:
            应变能 (J)
        """
        if self._K is None:
            self._K = self.structure.assemble_stiffness_matrix()
        
        energy = 0.5 * displacements @ self._K @ displacements
        logger.info(f"应变能: {energy:.4e} J")
        return energy
    
    def modal_analysis(self, num_modes: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """
        模态分析（求解固有频率和振型）
        
        Args:
            num_modes: 求解的模态数
            
        Returns:
            (固有频率数组, 振型矩阵)
        """
        from scipy.linalg import eigh
        
        logger.info(f"开始模态分析，求解 {num_modes} 阶模态...")
        
        K = self.structure.assemble_stiffness_matrix()
        M = self.structure.assemble_mass_matrix()
        
        free_dofs = self.structure.get_free_dofs()
        
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        M_ff = M[np.ix_(free_dofs, free_dofs)]
        
        # 求解广义特征值问题
        eigenvalues, eigenvectors = eigh(K_ff, M_ff)
        
        # 计算固有频率
        omega = np.sqrt(np.abs(eigenvalues))
        frequencies = omega / (2 * np.pi)
        
        # 取前 num_modes 阶
        num_modes = min(num_modes, len(frequencies))
        frequencies = frequencies[:num_modes]
        mode_shapes = eigenvectors[:, :num_modes]
        
        logger.info(f"前 {num_modes} 阶固有频率 (Hz): {frequencies}")
        
        return frequencies, mode_shapes
