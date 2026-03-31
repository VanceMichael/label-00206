"""自定义异常类"""


class FEMError(Exception):
    """有限元分析基础异常"""
    pass


class InputError(FEMError):
    """输入数据错误"""
    pass


class ValidationError(FEMError):
    """数据验证错误"""
    pass


class SolverError(FEMError):
    """求解器错误"""
    pass


class StructureError(FEMError):
    """结构定义错误"""
    pass


class BoundaryError(FEMError):
    """边界条件错误"""
    pass
