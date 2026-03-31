"""Flask Web 应用"""
import os
# 设置 Web 模式环境变量，用于 plotter 模块判断后端
os.environ['FEM_TRUSS_WEB_MODE'] = '1'

import json
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import numpy as np

from ..core import TrussStructure, Material
from ..solver import StaticSolver, DynamicSolver
from ..visualization import TrussPlotter
from ..io import DataReader, ResultWriter
from ..utils import get_logger, FEMError

logger = get_logger(__name__)

app = Flask(__name__,
           template_folder=Path(__file__).parent / 'templates',
           static_folder=Path(__file__).parent / 'static')

# 配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
# 使用绝对路径确保文件能正确访问
BASE_DIR = Path(__file__).parent.parent.parent.resolve()
app.config['UPLOAD_FOLDER'] = BASE_DIR / 'uploads'
app.config['OUTPUT_FOLDER'] = BASE_DIR / 'output'

# 确保目录存在
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(exist_ok=True)

# 全局结构对象
current_structure: TrussStructure = None


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'healthy'})


@app.route('/api/structure', methods=['POST'])
def create_structure():
    """创建结构"""
    global current_structure
    
    try:
        data = request.get_json()
        
        current_structure = TrussStructure()
        
        # 添加材料
        for mat in data.get('materials', []):
            material = Material(
                id=mat['id'],
                name=mat.get('name', f"Material_{mat['id']}"),
                E=float(mat['E']),
                nu=float(mat.get('nu', 0.3)),
                rho=float(mat.get('rho', 7850))
            )
            current_structure.add_material(material)
        
        # 添加节点
        for node in data.get('nodes', []):
            current_structure.add_node(
                id=node['id'],
                x=float(node['x']),
                y=float(node['y'])
            )
        
        # 添加单元
        for elem in data.get('elements', []):
            current_structure.add_element(
                id=elem['id'],
                node_i=elem['node_i'],
                node_j=elem['node_j'],
                material_id=elem.get('material_id', 1),
                area=float(elem['area'])
            )
        
        # 添加荷载
        for load in data.get('loads', []):
            current_structure.apply_load(
                node_id=load['node_id'],
                fx=float(load.get('fx', 0)),
                fy=float(load.get('fy', 0))
            )
        
        # 添加边界条件
        for bc in data.get('boundaries', []):
            current_structure.apply_boundary(
                node_id=bc['node_id'],
                fix_x=bc.get('fix_x', True),
                fix_y=bc.get('fix_y', True)
            )
        
        logger.info(f"结构创建成功:\n{current_structure.summary()}")
        
        return jsonify({
            'success': True,
            'message': '结构创建成功',
            'summary': {
                'nodes': current_structure.num_nodes,
                'elements': current_structure.num_elements,
                'ndof': current_structure.ndof
            }
        })
        
    except Exception as e:
        logger.error(f"创建结构失败: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/analyze/static', methods=['POST'])
def analyze_static():
    """执行静力分析"""
    global current_structure
    
    if current_structure is None:
        return jsonify({'success': False, 'error': '请先创建结构'}), 400
    
    try:
        solver = StaticSolver(current_structure)
        result = solver.solve()
        
        # 保存结果
        writer = ResultWriter(str(app.config['OUTPUT_FOLDER']))
        node_ids = sorted(current_structure.nodes.keys())
        writer.write_displacements(result.displacements, node_ids)
        writer.write_stresses(result.stresses, result.strains)
        
        # 生成可视化
        plotter = TrussPlotter(current_structure, str(app.config['OUTPUT_FOLDER']))
        plotter.plot_structure()
        plotter.plot_deformation(result.displacements)
        plotter.plot_stress_contour(result.stresses)
        
        # 准备返回数据
        displacements_list = []
        for i, node_id in enumerate(node_ids):
            displacements_list.append({
                'node_id': node_id,
                'ux': float(result.displacements[2*i]),
                'uy': float(result.displacements[2*i + 1])
            })
        
        stresses_list = [
            {'element_id': k, 'stress': float(v)}
            for k, v in result.stresses.items()
        ]
        
        return jsonify({
            'success': True,
            'message': '静力分析完成',
            'results': {
                'displacements': displacements_list,
                'stresses': stresses_list,
                'max_displacement': float(np.max(np.abs(result.displacements))),
                'max_stress': float(max(abs(s) for s in result.stresses.values()))
            },
            'files': {
                'displacements': 'displacements.csv',
                'stresses': 'stresses.csv',
                'structure_plot': 'structure.png',
                'deformation_plot': 'deformation.png',
                'stress_plot': 'stress_contour.png'
            }
        })
        
    except Exception as e:
        logger.error(f"静力分析失败: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analyze/dynamic', methods=['POST'])
def analyze_dynamic():
    """执行动力分析"""
    global current_structure
    
    if current_structure is None:
        return jsonify({'success': False, 'error': '请先创建结构'}), 400
    
    try:
        data = request.get_json() or {}
        
        # 获取参数
        duration = float(data.get('duration', 2.0))
        dt = float(data.get('dt', 0.01))
        damping_ratio = float(data.get('damping_ratio', 0.05))
        amplitude = float(data.get('amplitude', 0.1))  # 地震加速度幅值 (g)
        frequency = float(data.get('frequency', 2.0))  # 主频 (Hz)
        
        # 生成简谐地震波
        time = np.arange(0, duration, dt)
        acceleration = amplitude * 9.81 * np.sin(2 * np.pi * frequency * time)
        
        # 添加衰减
        decay = np.exp(-0.5 * time)
        acceleration = acceleration * decay
        
        # 求解
        solver = DynamicSolver(current_structure)
        solver.set_damping_ratio(damping_ratio)
        result = solver.solve(acceleration, time, direction='x')
        
        # 生成动画
        plotter = TrussPlotter(current_structure, str(app.config['OUTPUT_FOLDER']))
        plotter.create_animation(result)
        
        # 保存时程数据
        writer = ResultWriter(str(app.config['OUTPUT_FOLDER']))
        node_indices = list(range(current_structure.num_nodes))
        writer.write_time_history(result, node_indices[:3])  # 前3个节点
        
        return jsonify({
            'success': True,
            'message': '动力分析完成',
            'results': {
                'duration': duration,
                'time_steps': len(time),
                'max_displacement': float(np.max(np.abs(result.displacement_history)))
            },
            'files': {
                'animation': 'animation.gif',
                'time_history': 'time_history.csv'
            }
        })
        
    except Exception as e:
        logger.error(f"动力分析失败: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/modal', methods=['POST'])
def modal_analysis():
    """模态分析"""
    global current_structure
    
    if current_structure is None:
        return jsonify({'success': False, 'error': '请先创建结构'}), 400
    
    try:
        data = request.get_json() or {}
        num_modes = int(data.get('num_modes', 6))
        
        solver = StaticSolver(current_structure)
        frequencies, mode_shapes = solver.modal_analysis(num_modes)
        
        return jsonify({
            'success': True,
            'message': '模态分析完成',
            'results': {
                'frequencies': frequencies.tolist(),
                'periods': (1 / frequencies).tolist()
            }
        })
        
    except Exception as e:
        logger.error(f"模态分析失败: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """下载结果文件"""
    try:
        return send_from_directory(
            app.config['OUTPUT_FOLDER'],
            filename,
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404


@app.route('/api/output/<filename>')
def get_output_file(filename):
    """获取输出文件（用于显示图片）"""
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404


@app.route('/api/examples/<name>')
def load_example(name):
    """加载示例结构"""
    examples = {
        'simple_truss': {
            'materials': [{'id': 1, 'name': 'Steel', 'E': 2.06e11, 'rho': 7850}],
            'nodes': [
                {'id': 1, 'x': 0, 'y': 0},
                {'id': 2, 'x': 4, 'y': 0},
                {'id': 3, 'x': 2, 'y': 2}
            ],
            'elements': [
                {'id': 1, 'node_i': 1, 'node_j': 2, 'material_id': 1, 'area': 0.001},
                {'id': 2, 'node_i': 1, 'node_j': 3, 'material_id': 1, 'area': 0.001},
                {'id': 3, 'node_i': 2, 'node_j': 3, 'material_id': 1, 'area': 0.001}
            ],
            'loads': [{'node_id': 3, 'fx': 0, 'fy': -10000}],
            'boundaries': [
                {'node_id': 1, 'fix_x': True, 'fix_y': True},
                {'node_id': 2, 'fix_x': False, 'fix_y': True}
            ]
        },
        'frame_truss': {
            'materials': [{'id': 1, 'name': 'Steel', 'E': 2.06e11, 'rho': 7850}],
            'nodes': [
                {'id': 1, 'x': 0, 'y': 0},
                {'id': 2, 'x': 3, 'y': 0},
                {'id': 3, 'x': 6, 'y': 0},
                {'id': 4, 'x': 0, 'y': 3},
                {'id': 5, 'x': 3, 'y': 3},
                {'id': 6, 'x': 6, 'y': 3}
            ],
            'elements': [
                {'id': 1, 'node_i': 1, 'node_j': 2, 'material_id': 1, 'area': 0.002},
                {'id': 2, 'node_i': 2, 'node_j': 3, 'material_id': 1, 'area': 0.002},
                {'id': 3, 'node_i': 4, 'node_j': 5, 'material_id': 1, 'area': 0.002},
                {'id': 4, 'node_i': 5, 'node_j': 6, 'material_id': 1, 'area': 0.002},
                {'id': 5, 'node_i': 1, 'node_j': 4, 'material_id': 1, 'area': 0.002},
                {'id': 6, 'node_i': 2, 'node_j': 5, 'material_id': 1, 'area': 0.002},
                {'id': 7, 'node_i': 3, 'node_j': 6, 'material_id': 1, 'area': 0.002},
                {'id': 8, 'node_i': 1, 'node_j': 5, 'material_id': 1, 'area': 0.0015},
                {'id': 9, 'node_i': 2, 'node_j': 4, 'material_id': 1, 'area': 0.0015},
                {'id': 10, 'node_i': 2, 'node_j': 6, 'material_id': 1, 'area': 0.0015},
                {'id': 11, 'node_i': 3, 'node_j': 5, 'material_id': 1, 'area': 0.0015}
            ],
            'loads': [
                {'node_id': 4, 'fx': 5000, 'fy': 0},
                {'node_id': 5, 'fx': 0, 'fy': -20000},
                {'node_id': 6, 'fx': 5000, 'fy': 0}
            ],
            'boundaries': [
                {'node_id': 1, 'fix_x': True, 'fix_y': True},
                {'node_id': 3, 'fix_x': False, 'fix_y': True}
            ]
        }
    }
    
    if name not in examples:
        return jsonify({'success': False, 'error': f'示例 {name} 不存在'}), 404
    
    return jsonify({'success': True, 'data': examples[name]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
