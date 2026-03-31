"""桁架结构类定义

支持两种单元类型：
1. TrussElement - 桁架杆单元（轴向拉压杆）
2. CSTElement - CST 平面应力三角形单元

两种单元可以混合使用，适用于不同的分析场景。
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .element import TrussElement
from .cst_element import CSTElement
from .material import Material
from ..utils import get_logger, StructureError, BoundaryError

logger = get_logger(__name__)


@dataclass
class Node:
    """节点类"""
    id: int
    x: float
    y: float
    
    @property
    def coords(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Load:
    """荷载类"""
    id: int
    node_id: int
    fx: float = 0.0  # X方向力
    fy: float = 0.0  # Y方向力


@dataclass
class Boundary:
    """边界条件类"""
    id: int
    node_id: int
    fix_x: bool = False  # X方向约束
    fix_y: bool = False  # Y方向约束
    
    @property
    def is_fixed(self) -> bool:
        return self.fix_x or self.fix_y


class TrussStructure:
    """
    平面桁架结构类
    
    管理节点、单元、荷载、边界条件，并提供整体矩阵组装功能。
    支持桁架杆单元（TrussElement）和 CST 三角形单元（CSTElement）。
    """
    
    def __init__(self, use_sparse: bool = False):
        """
        初始化结构
        
        Args:
            use_sparse: 是否使用稀疏矩阵存储（推荐节点数 > 500 时启用）
        """
        self.nodes: Dict[int, Node] = {}
        self.elements: Dict[int, TrussElement] = {}
        self.cst_elements: Dict[int, CSTElement] = {}  # CST 单元存储
        self.materials: Dict[int, Material] = {}
        self.loads: List[Load] = []
        self.boundaries: List[Boundary] = []
        
        self._ndof: int = 0  # 总自由度数
        self._dof_map: Dict[int, Tuple[int, int]] = {}  # 节点ID -> (dof_x, dof_y)
        self._use_sparse: bool = use_sparse  # 稀疏矩阵标志
        
        logger.info(f"创建新的桁架结构实例 (稀疏矩阵: {use_sparse})")
    
    def add_material(self, material: Material):
        """添加材料到材料库"""
        self.materials[material.id] = material
        logger.info(f"添加材料: {material}")
    
    def add_node(self, id: int, x: float, y: float) -> Node:
        """
        添加节点
        
        Args:
            id: 节点编号
            x: X坐标
            y: Y坐标
            
        Returns:
            创建的节点对象
        """
        if id in self.nodes:
            raise StructureError(f"节点 {id} 已存在")
        
        node = Node(id=id, x=x, y=y)
        self.nodes[id] = node
        self._update_dof_map()
        
        logger.debug(f"添加节点: {node}")
        return node
    
    def add_element(self, id: int, node_i: int, node_j: int, 
                    material_id: int, area: float) -> TrussElement:
        """
        添加单元
        
        Args:
            id: 单元编号
            node_i: 起始节点编号
            node_j: 终止节点编号
            material_id: 材料编号
            area: 截面面积
            
        Returns:
            创建的单元对象
        """
        if id in self.elements:
            raise StructureError(f"单元 {id} 已存在")
        if node_i not in self.nodes:
            raise StructureError(f"节点 {node_i} 不存在")
        if node_j not in self.nodes:
            raise StructureError(f"节点 {node_j} 不存在")
        if material_id not in self.materials:
            raise StructureError(f"材料 {material_id} 不存在")
        
        ni = self.nodes[node_i]
        nj = self.nodes[node_j]
        
        element = TrussElement(
            id=id,
            node_i=node_i,
            node_j=node_j,
            material=self.materials[material_id],
            area=area,
            coords_i=ni.coords,
            coords_j=nj.coords
        )
        
        self.elements[id] = element
        logger.debug(f"添加单元: {element}")
        return element
    
    def remove_node(self, id: int):
        """删除节点"""
        if id not in self.nodes:
            raise StructureError(f"节点 {id} 不存在")
        
        # 检查是否有单元连接到此节点
        for elem in self.elements.values():
            if elem.node_i == id or elem.node_j == id:
                raise StructureError(f"节点 {id} 被单元 {elem.id} 使用，无法删除")
        
        del self.nodes[id]
        self._update_dof_map()
        logger.info(f"删除节点 {id}")
    
    def remove_element(self, id: int):
        """删除单元"""
        if id not in self.elements:
            raise StructureError(f"单元 {id} 不存在")
        
        del self.elements[id]
        logger.info(f"删除单元 {id}")
    
    def add_cst_element(self, id: int, node_i: int, node_j: int, node_k: int,
                        material_id: int, thickness: float) -> CSTElement:
        """
        添加 CST 三角形单元
        
        Args:
            id: 单元编号
            node_i, node_j, node_k: 三个节点编号（逆时针排列）
            material_id: 材料编号
            thickness: 单元厚度 (m)
            
        Returns:
            创建的 CST 单元对象
        """
        if id in self.cst_elements:
            raise StructureError(f"CST 单元 {id} 已存在")
        for nid in [node_i, node_j, node_k]:
            if nid not in self.nodes:
                raise StructureError(f"节点 {nid} 不存在")
        if material_id not in self.materials:
            raise StructureError(f"材料 {material_id} 不存在")
        
        ni = self.nodes[node_i]
        nj = self.nodes[node_j]
        nk = self.nodes[node_k]
        
        element = CSTElement(
            id=id,
            node_ids=(node_i, node_j, node_k),
            material=self.materials[material_id],
            thickness=thickness,
            coords=[ni.coords, nj.coords, nk.coords]
        )
        
        self.cst_elements[id] = element
        logger.debug(f"添加 CST 单元: {element}")
        return element
    
    def remove_cst_element(self, id: int):
        """删除 CST 单元"""
        if id not in self.cst_elements:
            raise StructureError(f"CST 单元 {id} 不存在")
        
        del self.cst_elements[id]
        logger.info(f"删除 CST 单元 {id}")
    
    def apply_load(self, node_id: int, fx: float = 0.0, fy: float = 0.0) -> Load:
        """
        施加节点荷载
        
        Args:
            node_id: 节点编号
            fx: X方向力
            fy: Y方向力
            
        Returns:
            创建的荷载对象
        """
        if node_id not in self.nodes:
            raise StructureError(f"节点 {node_id} 不存在")
        
        load = Load(id=len(self.loads) + 1, node_id=node_id, fx=fx, fy=fy)
        self.loads.append(load)
        
        logger.info(f"施加荷载: 节点{node_id}, Fx={fx}, Fy={fy}")
        return load
    
    def apply_boundary(self, node_id: int, fix_x: bool = False, fix_y: bool = False) -> Boundary:
        """
        施加边界条件
        
        Args:
            node_id: 节点编号
            fix_x: 是否约束X方向
            fix_y: 是否约束Y方向
            
        Returns:
            创建的边界条件对象
        """
        if node_id not in self.nodes:
            raise StructureError(f"节点 {node_id} 不存在")
        
        boundary = Boundary(
            id=len(self.boundaries) + 1,
            node_id=node_id,
            fix_x=fix_x,
            fix_y=fix_y
        )
        self.boundaries.append(boundary)
        
        logger.info(f"施加边界条件: 节点{node_id}, fix_x={fix_x}, fix_y={fix_y}")
        return boundary
    
    def _update_dof_map(self):
        """更新自由度映射"""
        self._dof_map.clear()
        sorted_nodes = sorted(self.nodes.keys())
        
        for i, node_id in enumerate(sorted_nodes):
            self._dof_map[node_id] = (2 * i, 2 * i + 1)
        
        self._ndof = 2 * len(self.nodes)
    
    @property
    def ndof(self) -> int:
        """总自由度数"""
        return self._ndof
    
    @property
    def num_nodes(self) -> int:
        """节点数量"""
        return len(self.nodes)
    
    @property
    def num_elements(self) -> int:
        """单元数量（桁架单元 + CST 单元）"""
        return len(self.elements) + len(self.cst_elements)
    
    @property
    def num_truss_elements(self) -> int:
        """桁架单元数量"""
        return len(self.elements)
    
    @property
    def num_cst_elements(self) -> int:
        """CST 单元数量"""
        return len(self.cst_elements)
    
    def get_dof_indices(self, node_id: int) -> Tuple[int, int]:
        """获取节点的自由度索引"""
        if node_id not in self._dof_map:
            raise StructureError(f"节点 {node_id} 不存在")
        return self._dof_map[node_id]
    
    def get_element_dof_indices(self, element: Union[TrussElement, CSTElement]) -> List[int]:
        """获取单元的自由度索引列表"""
        if isinstance(element, TrussElement):
            dof_i = self._dof_map[element.node_i]
            dof_j = self._dof_map[element.node_j]
            return [dof_i[0], dof_i[1], dof_j[0], dof_j[1]]
        elif isinstance(element, CSTElement):
            dofs = []
            for nid in element.node_ids:
                dof = self._dof_map[nid]
                dofs.extend([dof[0], dof[1]])
            return dofs
        else:
            raise StructureError(f"未知单元类型: {type(element)}")
    
    def assemble_stiffness_matrix(self) -> np.ndarray:
        """
        组装整体刚度矩阵
        
        支持稀疏矩阵存储以提高大规模问题的效率。
        
        Returns:
            ndof x ndof 整体刚度矩阵（密集或稀疏格式）
        """
        if self._ndof == 0:
            raise StructureError("结构中没有节点")
        
        if self._use_sparse:
            return self._assemble_stiffness_sparse()
        
        K = np.zeros((self._ndof, self._ndof))
        
        # 组装桁架单元
        for elem in self.elements.values():
            ke = elem.compute_stiffness_matrix()
            dof_indices = self.get_element_dof_indices(elem)
            
            for i, gi in enumerate(dof_indices):
                for j, gj in enumerate(dof_indices):
                    K[gi, gj] += ke[i, j]
        
        # 组装 CST 单元
        for elem in self.cst_elements.values():
            ke = elem.compute_stiffness_matrix()
            dof_indices = self.get_element_dof_indices(elem)
            
            for i, gi in enumerate(dof_indices):
                for j, gj in enumerate(dof_indices):
                    K[gi, gj] += ke[i, j]
        
        logger.info(f"整体刚度矩阵组装完成, 尺寸: {K.shape}")
        return K
    
    def _assemble_stiffness_sparse(self):
        """使用稀疏矩阵格式组装刚度矩阵"""
        from scipy.sparse import lil_matrix, csr_matrix
        
        K = lil_matrix((self._ndof, self._ndof))
        
        # 组装桁架单元
        for elem in self.elements.values():
            ke = elem.compute_stiffness_matrix()
            dof_indices = self.get_element_dof_indices(elem)
            
            for i, gi in enumerate(dof_indices):
                for j, gj in enumerate(dof_indices):
                    K[gi, gj] += ke[i, j]
        
        # 组装 CST 单元
        for elem in self.cst_elements.values():
            ke = elem.compute_stiffness_matrix()
            dof_indices = self.get_element_dof_indices(elem)
            
            for i, gi in enumerate(dof_indices):
                for j, gj in enumerate(dof_indices):
                    K[gi, gj] += ke[i, j]
        
        K_csr = csr_matrix(K)
        logger.info(f"稀疏刚度矩阵组装完成, 尺寸: {K_csr.shape}, 非零元素: {K_csr.nnz}")
        return K_csr
    
    def assemble_mass_matrix(self, consistent: bool = True) -> np.ndarray:
        """
        组装整体质量矩阵
        
        Args:
            consistent: 是否使用一致质量矩阵
            
        Returns:
            ndof x ndof 整体质量矩阵
        """
        if self._ndof == 0:
            raise StructureError("结构中没有节点")
        
        if self._use_sparse:
            return self._assemble_mass_sparse(consistent)
        
        M = np.zeros((self._ndof, self._ndof))
        
        # 组装桁架单元
        for elem in self.elements.values():
            me = elem.compute_mass_matrix(consistent=consistent)
            dof_indices = self.get_element_dof_indices(elem)
            
            for i, gi in enumerate(dof_indices):
                for j, gj in enumerate(dof_indices):
                    M[gi, gj] += me[i, j]
        
        # 组装 CST 单元
        for elem in self.cst_elements.values():
            me = elem.compute_mass_matrix(consistent=consistent)
            dof_indices = self.get_element_dof_indices(elem)
            
            for i, gi in enumerate(dof_indices):
                for j, gj in enumerate(dof_indices):
                    M[gi, gj] += me[i, j]
        
        logger.info(f"整体质量矩阵组装完成, 尺寸: {M.shape}")
        return M
    
    def _assemble_mass_sparse(self, consistent: bool = True):
        """使用稀疏矩阵格式组装质量矩阵"""
        from scipy.sparse import lil_matrix, csr_matrix
        
        M = lil_matrix((self._ndof, self._ndof))
        
        for elem in self.elements.values():
            me = elem.compute_mass_matrix(consistent=consistent)
            dof_indices = self.get_element_dof_indices(elem)
            
            for i, gi in enumerate(dof_indices):
                for j, gj in enumerate(dof_indices):
                    M[gi, gj] += me[i, j]
        
        for elem in self.cst_elements.values():
            me = elem.compute_mass_matrix(consistent=consistent)
            dof_indices = self.get_element_dof_indices(elem)
            
            for i, gi in enumerate(dof_indices):
                for j, gj in enumerate(dof_indices):
                    M[gi, gj] += me[i, j]
        
        return csr_matrix(M)
    
    def assemble_load_vector(self) -> np.ndarray:
        """
        组装荷载向量
        
        Returns:
            ndof x 1 荷载向量
        """
        F = np.zeros(self._ndof)
        
        for load in self.loads:
            dof_x, dof_y = self._dof_map[load.node_id]
            F[dof_x] += load.fx
            F[dof_y] += load.fy
        
        logger.info(f"荷载向量组装完成, 总荷载数: {len(self.loads)}")
        return F
    
    def get_constrained_dofs(self) -> List[int]:
        """获取被约束的自由度索引列表"""
        constrained = []
        
        for bc in self.boundaries:
            dof_x, dof_y = self._dof_map[bc.node_id]
            if bc.fix_x:
                constrained.append(dof_x)
            if bc.fix_y:
                constrained.append(dof_y)
        
        return sorted(set(constrained))
    
    def get_free_dofs(self) -> List[int]:
        """获取自由的自由度索引列表"""
        constrained = set(self.get_constrained_dofs())
        return [i for i in range(self._ndof) if i not in constrained]
    
    def validate(self) -> bool:
        """
        验证结构定义的完整性
        
        Returns:
            验证是否通过
        """
        errors = []
        
        # 检查节点
        if len(self.nodes) < 2:
            errors.append("至少需要2个节点")
        
        # 检查单元
        if len(self.elements) + len(self.cst_elements) < 1:
            errors.append("至少需要1个单元")
        
        # 检查边界条件
        constrained_dofs = self.get_constrained_dofs()
        if len(constrained_dofs) < 3:
            errors.append("边界条件不足，结构可能不稳定（至少需要约束3个自由度）")
        
        # 检查荷载
        if len(self.loads) == 0:
            errors.append("未施加任何荷载")
        
        if errors:
            for err in errors:
                logger.warning(f"结构验证警告: {err}")
            return False
        
        logger.info("结构验证通过")
        return True
    
    def summary(self) -> str:
        """生成结构摘要信息"""
        lines = [
            "=" * 50,
            "桁架结构摘要",
            "=" * 50,
            f"节点数量: {self.num_nodes}",
            f"桁架单元数量: {self.num_truss_elements}",
            f"CST 单元数量: {self.num_cst_elements}",
            f"材料种类: {len(self.materials)}",
            f"荷载数量: {len(self.loads)}",
            f"边界条件: {len(self.boundaries)}",
            f"总自由度: {self._ndof}",
            f"约束自由度: {len(self.get_constrained_dofs())}",
            f"稀疏矩阵模式: {'启用' if self._use_sparse else '禁用'}",
            "=" * 50
        ]
        return "\n".join(lines)
    
    def solve_static(self) -> 'StaticResult':
        """
        求解静力响应
        
        按照 Prompt 要求，TrussStructure 类应包含求解方法。
        此方法封装了 StaticSolver 的调用。
        
        Returns:
            StaticResult 包含位移、应力、应变、反力
        """
        from ..solver.static import StaticSolver
        from ..io.writer import StaticResult
        from scipy.linalg import solve
        
        logger.info("开始静力分析...")
        
        if not self.validate():
            logger.warning("结构验证未完全通过，继续求解...")
        
        # 组装矩阵
        K = self.assemble_stiffness_matrix()
        F = self.assemble_load_vector()
        
        # 获取自由度信息
        free_dofs = self.get_free_dofs()
        constrained_dofs = self.get_constrained_dofs()
        
        if len(free_dofs) == 0:
            raise StructureError("所有自由度都被约束，无法求解")
        
        # 提取自由自由度的子矩阵
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        
        # 求解位移
        u_f = solve(K_ff, F_f)
        
        # 组装完整位移向量
        displacements = np.zeros(self._ndof)
        displacements[free_dofs] = u_f
        
        # 计算支座反力
        reactions = K @ displacements - F
        
        # 计算单元应力和应变
        stresses = {}
        strains = {}
        for elem_id, elem in self.elements.items():
            dof_indices = self.get_element_dof_indices(elem)
            elem_disp = displacements[dof_indices]
            stresses[elem_id] = elem.compute_stress(elem_disp)
            strains[elem_id] = elem.compute_strain(elem_disp)
        
        logger.info("静力分析完成")
        
        return StaticResult(
            displacements=displacements,
            stresses=stresses,
            strains=strains,
            reactions=reactions
        )
    
    def solve_dynamic(self, acceleration: np.ndarray, time: np.ndarray,
                      direction: str = 'x', damping_ratio: float = 0.05) -> 'DynamicResult':
        """
        求解动力响应
        
        按照 Prompt 要求，TrussStructure 类应包含求解方法。
        此方法封装了 DynamicSolver 的调用。
        
        Args:
            acceleration: 地震加速度时程 (m/s²)
            time: 时间序列 (s)
            direction: 地震作用方向 ('x' 或 'y')
            damping_ratio: 阻尼比
            
        Returns:
            DynamicResult 包含位移、速度、加速度时程
        """
        from ..solver.dynamic import DynamicSolver
        
        logger.info("开始动力分析...")
        
        solver = DynamicSolver(self)
        solver.set_damping_ratio(damping_ratio)
        result = solver.solve(acceleration, time, direction)
        
        logger.info("动力分析完成")
        return result
