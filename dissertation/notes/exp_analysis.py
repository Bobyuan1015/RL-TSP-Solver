

import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple, Any, Optional
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error
import os
from itertools import combinations
import colorsys
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ColorGenerator:
    """动态颜色生成器"""

    @staticmethod
    def generate_colors(n_colors: int, color_type: str = 'qualitative') -> List[str]:
        """
        生成指定数量和类型的颜色

        Args:
            n_colors: 需要的颜色数量
            color_type: 颜色类型 ('qualitative', 'sequential', 'diverging')

        Returns:
            颜色列表
        """
        if n_colors == 0:
            return []

        if color_type == 'qualitative':
            # 高对比度颜色调色板
            high_contrast_colors = [
                '#FF0000', '#0000FF', '#00FF00', '#FF8C00', '#800080',
                '#FF1493', '#00CED1', '#FFD700', '#8B4513', '#2E8B57',
                '#4169E1', '#DC143C', '#32CD32', '#FF4500', '#9932CC',
                '#00FFFF', '#FF69B4', '#8FBC8F', '#B22222', '#00FF7F'
            ]

            if n_colors <= len(high_contrast_colors):
                return high_contrast_colors[:n_colors]

            # 使用HSV色彩空间生成更多颜色
            colors = high_contrast_colors.copy()
            remaining = n_colors - len(colors)
            for i in range(remaining):
                hue = (i * 137.508) % 360  # 黄金角度
                saturation = 0.8 + 0.2 * (i % 2)
                value = 0.7 + 0.3 * ((i // 2) % 2)
                rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
                colors.append(hex_color)
            return colors[:n_colors]

        elif color_type == 'sequential':
            return sns.color_palette("viridis", n_colors).as_hex()
        elif color_type == 'diverging':
            return sns.color_palette("RdBu_r", n_colors).as_hex()
        else:
            return ColorGenerator.generate_colors(n_colors, 'qualitative')


# 计算optimality_gap函数
def calculate_optimality_gap(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['optimality_gap'] = (data['current_distance'] - data['optimal_distance']) * 100 / data['optimal_distance']
    return data


def process_stage0_data(df: pd.DataFrame, target_metric: str = 'optimality_gap', train_test: str = 'train'):
    """
    阶段0数据处理：保留run_id，对比对象聚合
    
    Args:
        df: 输入的pandas DataFrame
        target_metric: 对比对象 ('optimality_gap', 'total_reward')
        train_test: 训练/测试阶段
    
    Returns:
        stage0_data: 处理后的阶段0数据
    """
    print(f"处理阶段0数据，目标指标: {target_metric}, 阶段: {train_test}")
    
    # 筛选done=1的数据
    filtered_data = df[(df['done'] == 1) & (df['train_test'] == train_test)].copy()
    
    # 计算optimality_gap
    if target_metric == 'optimality_gap':
        filtered_data = calculate_optimality_gap(filtered_data)

    # 以指定维度为单位进行聚合
    group_cols = ['algorithm', 'city_num', 'mode', 'state_type', 'train_test', 'instance_id', 'run_id']
    
    if target_metric == 'optimality_gap':
        agg_func = 'min'  # optimality_gap取最小值
    else:
        agg_func = 'max'  # 其他指标如total_reward取最大值
    
    stage0_data = filtered_data.groupby(group_cols).agg({
        target_metric: agg_func
    }).reset_index()
    
    # 重命名聚合后的列
    stage0_data.rename(columns={target_metric: 'metric'}, inplace=True)
    stage0_data['metric_type'] = target_metric
    
    print(f"阶段0数据处理完成，形状: {stage0_data.shape}")
    return stage0_data


def process_stage1_data(stage0_data: pd.DataFrame):
    """
    阶段1数据处理：消去run_id，对metric求聚合
    
    Args:
        stage0_data: 阶段0处理后的数据
    
    Returns:
        stage1_data: 处理后的阶段1数据
    """
    print("处理阶段1数据...")
    
    group_cols = ['algorithm', 'city_num', 'mode', 'state_type', 'train_test', 'instance_id', 'metric_type']
    
    stage1_data = stage0_data.groupby(group_cols).agg({
        'metric': ['max', 'min', 'mean', 'std', 'count']
    }).reset_index()
    
    # 重新命名列
    stage1_data.columns = group_cols + ['runs_max', 'runs_min', 'runs_mean', 'runs_std', 'runs_count']
    
    print(f"阶段1数据处理完成，形状: {stage1_data.shape}")
    return stage1_data


def process_stage2_data(stage1_data: pd.DataFrame):
    """
    阶段2数据处理：消去instance_id，对runs_xxx求聚合
    
    Args:
        stage1_data: 阶段1处理后的数据
    
    Returns:
        stage2_data: 处理后的阶段2数据
    """
    print("处理阶段2数据...")
    
    group_cols = ['algorithm', 'city_num', 'mode', 'state_type', 'train_test', 'metric_type']
    
    agg_dict = {}
    for metric in ['runs_max', 'runs_min', 'runs_mean']:
        agg_dict[metric] = ['max', 'min', 'mean', 'std']
    
    stage2_data = stage1_data.groupby(group_cols).agg(agg_dict).reset_index()
    
    # 重新命名列
    new_columns = group_cols.copy()
    for metric in ['runs_max', 'runs_min', 'runs_mean']:
        for stat in ['max', 'min', 'mean', 'std']:
            new_columns.append(f'instances_{metric}_{stat}')
    
    stage2_data.columns = new_columns
    
    print(f"阶段2数据处理完成，形状: {stage2_data.shape}")
    return stage2_data


def calculate_component_contributions(stage2_data: pd.DataFrame, runs_dimension: str = 'mean', performance_better_when: str = 'smaller'):
    """
    计算组件贡献度分析和消融路径分析 - 基于消融实验理论
    
    Args:
        stage2_data: 阶段1处理后的数据
        runs_dimension: 使用的runs维度 ('max', 'min', 'mean') 
        performance_better_when: 'smaller'表示越小越好，'larger'表示越大越好
    
    Returns:
        tuple: (contributions, pathway_analysis, performance_better_when)
            - contributions: 组件贡献度分析结果DataFrame
            - pathway_analysis: 消融路径分析结果DataFrame
            - performance_better_when: 实际使用的性能优化方向
    """
    print(f"执行组件贡献度分析和消融路径分析，使用runs_{runs_dimension}...")
    
    # 基础状态组件
    base_states = ["current_city_onehot", "visited_mask", "order_embedding", "distances_from_current"]
    
    # 将stage2_data转换为类似performance_metrics的格式
    metrics = stage2_data.copy()

    metric_col = f'instances_runs_{runs_dimension}_mean'
    
    # 根据metric_type自动确定performance_better_when
    if 'metric_type' in metrics.columns and metrics['metric_type'].iloc[0] == 'optimality_gap':
        performance_better_when = 'smaller'
    elif 'metric_type' in metrics.columns and 'reward' in metrics['metric_type'].iloc[0]:
        performance_better_when = 'larger'

    # 重命名列以匹配原始逻辑
    metrics['optimality_gap_mean'] = metrics[f'instances_runs_{runs_dimension}_mean']
    metrics['optimality_gap_std'] = metrics[f'instances_runs_{runs_dimension}_std']
    # metrics['optimality_gap_count'] = metrics[f'runs_count']
    
    contributions = []
    pathway_analysis = []
    
    for algorithm in metrics['algorithm'].unique():
        for city_num in metrics['city_num'].unique():
            for mode in metrics['mode'].unique():
                for train_test in metrics['train_test'].unique():
                    
                    subset = metrics[
                        (metrics['algorithm'] == algorithm) &
                        (metrics['city_num'] == city_num) &
                        (metrics['mode'] == mode) &
                        (metrics['train_test'] == train_test)
                    ]
                    
                    if len(subset) == 0:
                        continue
                    
                    full_row = subset[subset['state_type'] == 'full']
                    if len(full_row) == 0:
                        continue
                    
                    # full_stats = {
                    #     'mean': full_row.iloc[0]['optimality_gap_mean'],
                    #     'std': full_row.iloc[0]['optimality_gap_std'],
                    #     'count': full_row.iloc[0]['optimality_gap_count']
                    # }
                    
                    performance_dict = {}
                    for _, row in subset.iterrows():
                        state_type = row['state_type']
                        performance_dict[state_type] = row[metric_col]
                    
                    if 'full' not in performance_dict:
                        continue
                    
                    full_performance = performance_dict['full']
                    
                    # 计算各组件的边际贡献
                    component_contributions = _calculate_marginal_contributions(performance_dict, base_states, performance_better_when)
                    
                    # 计算交互效应
                    interaction_effects = _calculate_interaction_effects(performance_dict, base_states, performance_better_when)
                    
                    # 计算组件重要性排序
                    importance_ranking = _calculate_importance_ranking(performance_dict, base_states, performance_better_when)

                    # 组件贡献度结果
                    result = {
                        'algorithm': algorithm,
                        'city_num': city_num,
                        'mode': mode,
                        'train_test': train_test,
                        'runs_dimension': runs_dimension,
                        'full_performance': full_performance,
                        **performance_dict,
                        **component_contributions,
                        **interaction_effects,
                        **importance_ranking,
                    }
                    
                    contributions.append(result)
                    
                    # 计算消融路径分析
                    pathways = _analyze_ablation_pathways(performance_dict, performance_better_when)
                    
                    # 处理消融路径分析结果
                    for pathway_name, pathway_data in pathways.items():
                        
                        # 处理具体路径数据
                        pathway_performance = pathway_data.get('pathway_performance', [])  # 首元素为full状态的performance
                        degradation_rates = pathway_data.get('degradation_rate', [])
                        total_degradation = pathway_data.get('total_degradation', 0)
                        
                        # 计算路径特征指标
                        pathway_length = len(pathway_performance)
                        max_degradation = max(degradation_rates) if degradation_rates else 0
                        min_degradation = min(degradation_rates) if degradation_rates else 0
                        degradation_variance = np.var(degradation_rates) if degradation_rates else 0
                        
                        # 路径效率：总退化/路径长度
                        pathway_efficiency = abs(total_degradation) / max(pathway_length - 1,
                                                                          1) if pathway_length > 1 else 0
                        
                        pathway_result = {
                            'algorithm': algorithm,
                            'city_num': city_num,
                            'mode': mode,
                            'train_test': train_test,
                            'runs_dimension': runs_dimension,
                            'pathway_name': pathway_name,
                            'pathway_type': 'ablation_sequence',
                            'pathway_length': pathway_length,
                            'total_degradation': total_degradation,
                            'max_single_step_degradation': max_degradation,
                            'min_single_step_degradation': min_degradation,
                            'degradation_variance': degradation_variance,
                            'pathway_efficiency': pathway_efficiency,
                            'pathway_performance_list': str(pathway_performance),  # 转为字符串存储
                            'degradation_rate_list': str(degradation_rates),
                            'pathway_description': pathway_data.get('pathway_description', ''),
                            # 添加统计信息的默认值
                            'num_available_combinations': 0,
                            'max_components_removed': 0,
                            'average_single_step_degradation': np.mean(
                                degradation_rates) if degradation_rates else 0
                        }
                        
                        # 如果有组件信息，添加组件分析
                        if 'pathway_components' in pathway_data:
                            components_info = pathway_data['pathway_components']
                            pathway_result['pathway_components'] = str(components_info)
                            
                            # 分析组件移除模式
                            if components_info:
                                # 计算每步新增移除的组件数
                                step_removals = []
                                prev_count = 0
                                for step_components in components_info:
                                    current_count = len(step_components)
                                    step_removals.append(current_count - prev_count)
                                    prev_count = current_count
                                
                                pathway_result['removal_pattern'] = str(step_removals)
                                pathway_result['final_components_removed'] = len(
                                    components_info[-1]) if components_info else 0
                        else:
                            pathway_result['pathway_components'] = ''
                            pathway_result['removal_pattern'] = ''
                            pathway_result['final_components_removed'] = 0
                        
                        pathway_analysis.append(pathway_result)
    
    return pd.DataFrame(contributions), pd.DataFrame(pathway_analysis), performance_better_when


def _calculate_marginal_contributions(performance_dict: Dict[str, float], base_states: List[str], performance_better_when: str = 'smaller') -> Dict[str, float]:
    """计算各组件的边际贡献"""
    contributions = {}
    full_perf = performance_dict.get('full', 0)
    
    # 根据优化方向确定"更好"的含义
    if performance_better_when == 'smaller':
        # 对于optimality_gap等指标，越小越好
        def get_degradation(current, baseline):
            return current - baseline  # 正值表示性能变差
    else:
        # 对于reward等指标，越大越好
        def get_degradation(current, baseline):
            return baseline - current  # 正值表示性能变差
    
    # 计算单组件移除的影响
    for component in base_states:
        remove_key = f'ablation_remove_{component.split("_")[0]}'
        if remove_key in performance_dict:
            # 边际贡献 = 移除该组件后的性能下降
            contribution = get_degradation(performance_dict[remove_key], full_perf)
            contributions[f'{component}_marginal_contribution'] = contribution
        else:
            contributions[f'{component}_marginal_contribution'] = 0.0
    
    return contributions


def _calculate_interaction_effects(performance_dict: Dict[str, float], base_states: List[str], performance_better_when: str = 'smaller') -> Dict[str, float]:
    """计算组件间的交互效应"""
    # 计算交互效应:  该函数计算当同时移除两个组件时，产生的交互效应是否大于、小于或等于单独移除这两个组件效应的简单相加。
    # 哪些状态组件必须同时保留（正交互效应）
    # 哪些组件可能功能重复（负交互效应）
    # 哪些组件相互独立（零交互效应）
    interactions = {}
    full_perf = performance_dict.get('full', 0)
    
    # 根据优化方向确定"更好"的含义
    if performance_better_when == 'smaller':
        # 对于optimality_gap等指标，越小越好
        def get_degradation(current, baseline):
            return current - baseline  # 正值表示性能变差
    else:
        # 对于reward等指标，越大越好
        def get_degradation(current, baseline):
            return baseline - current  # 正值表示性能变差
    
    # 计算两两组件的交互效应
    for i, comp1 in enumerate(base_states):
        for j, comp2 in enumerate(base_states[i + 1:], i + 1):
            comp1_short = comp1.split("_")[0]
            comp2_short = comp2.split("_")[0]
            
            # 构建同时移除两个组件的状态键
            remove_both_key = f'ablation_remove_{comp1_short}_{comp2_short}'
            
            if remove_both_key in performance_dict:
                remove_comp1_key = f'ablation_remove_{comp1_short}'
                remove_comp2_key = f'ablation_remove_{comp2_short}'
                
                if remove_comp1_key in performance_dict and remove_comp2_key in performance_dict:
                    # 交互效应 = 同时移除的影响 - 单独移除的影响之和
                    both_removed = get_degradation(performance_dict[remove_both_key], full_perf)
                    comp1_removed = get_degradation(performance_dict[remove_comp1_key], full_perf)
                    comp2_removed = get_degradation(performance_dict[remove_comp2_key], full_perf)
                    
                    interaction = both_removed - (comp1_removed + comp2_removed)
                    interactions[f'{comp1_short}_{comp2_short}_interaction'] = interaction
    
    return interactions


def _calculate_importance_ranking(performance_dict: Dict[str, float], base_states: List[str], performance_better_when: str = 'smaller') -> Dict[str, any]:
    """
        计算组件重要性排序

        重要性度量：基于边际贡献的绝对值
        Importance_i = |MC_i| = |P(S \ {i}) - P(S)|

        排序规则：
        1. 边际贡献绝对值越大，重要性越高
        2. 同时考虑统计显著性
        3. 提供重要性等级分类

        例子：
        visited_mask: MC = 10% → 重要性rank = 1 (最重要)
        current_city: MC = 8% → 重要性rank = 2
        order_embedding: MC = 3% → 重要性rank = 3
        distances: MC = 1% → 重要性rank = 4 (最不重要)
        """
    full_perf = performance_dict.get('full', 0)
    component_impacts = {}
    
    # 根据优化方向确定"更好"的含义
    if performance_better_when == 'smaller':
        # 对于optimality_gap等指标，越小越好
        def get_degradation(current, baseline):
            return current - baseline  # 正值表示性能变差
    else:
        # 对于reward等指标，越大越好
        def get_degradation(current, baseline):
            return baseline - current  # 正值表示性能变差
    
    for component in base_states:
        remove_key = f'ablation_remove_{component.split("_")[0]}'
        if remove_key in performance_dict:
            impact = get_degradation(performance_dict[remove_key], full_perf)
            component_impacts[component] = impact
    
    # 按影响程度排序（影响越大越重要）
    sorted_components = sorted(component_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    
    ranking = {}
    for rank, (component, impact) in enumerate(sorted_components, 1):
        ranking[f'{component}_importance_rank'] = rank
        ranking[f'{component}_impact_magnitude'] = abs(impact)
    
    return ranking



def _analyze_ablation_pathways(performance_dict: Dict[str, float], 
                               performance_better_when: str = 'smaller') -> Dict[str, Dict]:
    """
    对比剔除状态个数的衰减（组合中最小衰减和最大衰减）
    
    Args:
        performance_dict: 性能字典
        performance_better_when: 'smaller'表示越小越好，'larger'表示越大越好
    """
    pathways = {}
    full_perf = performance_dict.get('full', 0)
    
    # 根据优化方向确定"更好"的含义
    if performance_better_when == 'smaller':
        # 对于optimality_gap等指标，越小越好
        def is_better(val1, val2):
            return val1 < val2
        
        def get_degradation(current, baseline):
            return current - baseline  # 正值表示性能变差
    else:
        # 对于reward等指标，越大越好
        def is_better(val1, val2):
            return val1 > val2
        
        def get_degradation(current, baseline):
            return baseline - current  # 正值表示性能变差
    
    # 构建基于实际数据的路径
    available_pathways = {}
    
    for state_key, performance in performance_dict.items():
        if state_key.startswith('ablation_remove_') and state_key != 'full':
            removed_components = state_key.replace('ablation_remove_', '').split('_')
            num_removed = len(removed_components)
            
            if num_removed not in available_pathways:
                available_pathways[num_removed] = []
            
            degradation = get_degradation(performance, full_perf)
            available_pathways[num_removed].append({
                'components': removed_components,
                'performance': performance,
                'degradation': degradation
            })
    
    # 构建最优路径（最小性能退化）
    actual_pathway_perf = [full_perf]
    actual_pathway_components = []
    
    worst_pathway_perf = [full_perf]
    worst_pathway_components = []
    
    for num_removed in sorted(available_pathways.keys()):  # remove状态，从少到多，性能逐渐退化
        # 选择性能退化最小的组合
        best_combination = min(available_pathways[num_removed],
                               key=lambda x: x['degradation'])
        actual_pathway_perf.append(best_combination['performance'])
        actual_pathway_components.append(best_combination['components'])
        
        # 构建最坏路径（最大性能退化）
        worst_combination = max(available_pathways[num_removed],
                                key=lambda x: x['degradation'])
        worst_pathway_perf.append(worst_combination['performance'])
        worst_pathway_components.append(worst_combination['components'])
    
    # 按照remove state组合个数为单位统计
    pathways['optimal_actual'] = {
        'pathway_performance': actual_pathway_perf,
        'total_degradation': get_degradation(actual_pathway_perf[-1], actual_pathway_perf[0]) if len(
            actual_pathway_perf) > 1 else 0,
        'degradation_rate': [get_degradation(actual_pathway_perf[i+1], actual_pathway_perf[i])
                             for i in range(len(actual_pathway_perf) - 1)],
        'pathway_components': actual_pathway_components,
        'pathway_description': f'Optimal path (minimal degradation, {performance_better_when} is better)'
    }
    
    # 按照remove state组合个数为单位统计
    pathways['worst_case'] = {
        'pathway_performance': worst_pathway_perf,
        'total_degradation': get_degradation(worst_pathway_perf[-1], worst_pathway_perf[0]) if len(
            worst_pathway_perf) > 1 else 0,
        'degradation_rate': [get_degradation(worst_pathway_perf[i+1], worst_pathway_perf[i])
                             for i in range(len(worst_pathway_perf) - 1)],
        'pathway_components': worst_pathway_components,
        'pathway_description': f'Worst case path (maximal degradation, {performance_better_when} is better)'
    }
    
    return pathways


def calculate_ablation_pathway_analysis(stage2_data: pd.DataFrame, performance_better_when='smaller') -> pd.DataFrame:
    """
    计算消融路径分析
    
    Args:
        stage1_data: 阶段1处理后的数据
        performance_better_when: 'smaller'表示越小越好，'larger'表示越大越好
    
    Returns:
        pathway_analysis: 路径分析结果DataFrame
    """
    print("执行消融路径分析...")
    
    pathway_analysis = []

    for algorithm in stage2_data['algorithm'].unique():
        for city_num in stage2_data['city_num'].unique():
            for mode in stage2_data['mode'].unique():
                for train_test in stage2_data['train_test'].unique():
                    
                    subset = stage2_data[
                        (stage2_data['algorithm'] == algorithm) &
                        (stage2_data['city_num'] == city_num) &
                        (stage2_data['mode'] == mode) &
                        (stage2_data['train_test'] == train_test)
                    ]
                    
                    if len(subset) == 0:
                        continue
                    
                    performance_dict = {}
                    for _, row in subset.iterrows():
                        performance_dict[row['state_type']] = row[metric_col]  # 使用metric_col作为性能指标
                    
                    if 'full' not in performance_dict:
                        continue
                    
                    # 对比剔除状态个数的衰减（组合中最小衰减和最大衰减）
                    pathways = _analyze_ablation_pathways(performance_dict, performance_better_when)
                    
                    # 处理新的路径结构
                    for pathway_name, pathway_data in pathways.items():
                        
                        # 处理具体路径数据
                        pathway_performance = pathway_data.get('pathway_performance', [])  # 首元素为full状态的performance
                        degradation_rates = pathway_data.get('degradation_rate', [])
                        total_degradation = pathway_data.get('total_degradation', 0)
                        
                        # 计算路径特征指标
                        pathway_length = len(pathway_performance)
                        max_degradation = max(degradation_rates) if degradation_rates else 0
                        min_degradation = min(degradation_rates) if degradation_rates else 0
                        degradation_variance = np.var(degradation_rates) if degradation_rates else 0
                        
                        # 路径效率：总退化/路径长度
                        pathway_efficiency = abs(total_degradation) / max(pathway_length - 1,
                                                                          1) if pathway_length > 1 else 0
                        
                        result = {
                            'algorithm': algorithm,
                            'city_num': city_num,
                            'mode': mode,
                            'train_test': train_test,
                            'pathway_name': pathway_name,
                            'pathway_type': 'ablation_sequence',
                            'pathway_length': pathway_length,
                            'total_degradation': total_degradation,
                            'max_single_step_degradation': max_degradation,
                            'min_single_step_degradation': min_degradation,
                            'degradation_variance': degradation_variance,
                            'pathway_efficiency': pathway_efficiency,
                            'pathway_performance_list': str(pathway_performance),  # 转为字符串存储
                            'degradation_rate_list': str(degradation_rates),
                            'pathway_description': pathway_data.get('pathway_description', ''),
                            # 添加统计信息的默认值
                            'num_available_combinations': 0,
                            'max_components_removed': 0,
                            'average_single_step_degradation': np.mean(
                                degradation_rates) if degradation_rates else 0
                        }
                        
                        # 如果有组件信息，添加组件分析
                        if 'pathway_components' in pathway_data:
                            components_info = pathway_data['pathway_components']
                            result['pathway_components'] = str(components_info)
                            
                            # 分析组件移除模式
                            if components_info:
                                # 计算每步新增移除的组件数
                                step_removals = []
                                prev_count = 0
                                for step_components in components_info:
                                    current_count = len(step_components)
                                    step_removals.append(current_count - prev_count)
                                    prev_count = current_count
                                
                                result['removal_pattern'] = str(step_removals)
                                result['final_components_removed'] = len(
                                    components_info[-1]) if components_info else 0
                        else:
                            result['pathway_components'] = ''
                            result['removal_pattern'] = ''
                            result['final_components_removed'] = 0
                        
                        pathway_analysis.append(result)
    
    return pd.DataFrame(pathway_analysis)


def _plot_degradation_from_pathway_data_for_group(ax, group_data, pathway_analysis):
    """为特定组合使用pathway_analysis数据绘制退化图"""
    try:
        if pathway_analysis is None or len(pathway_analysis) == 0:
            ax.text(0.5, 0.5, 'No pathway data available',
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        # 获取当前组合的标识信息
        group_info = group_data.iloc[0] if len(group_data) > 0 else None
        if group_info is None:
            ax.text(0.5, 0.5, 'No group data available',
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        # 筛选对应组合的有效路径数据
        valid_pathways = pathway_analysis[
            (pathway_analysis['algorithm'] == group_info['algorithm']) &
            (pathway_analysis['city_num'] == group_info['city_num']) &
            (pathway_analysis['mode'] == group_info['mode']) &
            (pathway_analysis['train_test'] == group_info['train_test']) &
            (pathway_analysis['pathway_type'] == 'ablation_sequence') &
            (pathway_analysis['pathway_length'] > 1) &
            (pathway_analysis['degradation_rate_list'].notna())
        ]
        
        if len(valid_pathways) == 0:
            ax.text(0.5, 0.5, 'No valid pathway data for degradation analysis',
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        # 提取退化数据
        all_degradation_data = []
        
        for _, row in valid_pathways.iterrows():
            try:
                degradation_rates = eval(row['degradation_rate_list']) if isinstance(row['degradation_rate_list'], str) else row['degradation_rate_list']
                
                for step, degradation in enumerate(degradation_rates, 1):
                    all_degradation_data.append({
                        'num_components_removed': step,
                        'performance_degradation': degradation,
                        'pathway_name': row['pathway_name']
                    })
            except Exception as e:
                print(f"Error parsing pathway data: {e}")
                continue
        
        if not all_degradation_data:
            ax.text(0.5, 0.5, 'No degradation data could be extracted',
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        # 转换为DataFrame并绘制
        deg_df = pd.DataFrame(all_degradation_data)
        
        # 按移除组件数量分组统计
        degradation_stats = deg_df.groupby('num_components_removed').agg({
            'performance_degradation': ['mean', 'std', 'count']
        }).round(3)
        
        degradation_stats.columns = ['mean_degradation', 'std_degradation', 'count']
        degradation_stats = degradation_stats.reset_index()
        
        # 动态生成颜色
        color_generator = ColorGenerator()
        colors = color_generator.generate_colors(1, 'qualitative')
        line_color = colors[0] if colors else 'red'
        
        # 绘制主趋势线
        x_values = degradation_stats['num_components_removed']
        y_values = degradation_stats['mean_degradation']
        y_errors = degradation_stats['std_degradation']
        
        ax.plot(x_values, y_values, 'o-', linewidth=3, markersize=8,
                color=line_color, label='Mean Degradation')
        
        # 添加误差棒
        ax.errorbar(x_values, y_values, yerr=y_errors,
                    capsize=3, capthick=1, alpha=0.7, color=line_color)
        
        # 添加数据点标签
        for x, y, count in zip(x_values, y_values, degradation_stats['count']):
            ax.annotate(f'{y:.2f}%\n(n={count})', (x, y),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Number of Components Removed')
        ax.set_ylabel('Performance Degradation (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
    except Exception as e:
        print(f"Error in degradation plotting: {e}")
        ax.text(0.5, 0.5, f'Plotting error: {str(e)[:50]}...',
                ha='center', va='center', transform=ax.transAxes)


# 定义独立的绘图函数
def plot_single_metric_chart(data, metric_col, chart_name, save_dir):
    plt.figure(figsize=(12, 8))
    color_generator = ColorGenerator()
    state_types = data['state_type'].unique()
    colors = color_generator.generate_colors(len(state_types))

    for i, state_type in enumerate(state_types):
        subset = data[data['state_type'] == state_type]
        if len(subset) > 0:
            plt.plot(subset['instance_id'], subset[metric_col],
                     'o-', label=state_type, color=colors[i], linewidth=2, markersize=6)

    plt.xlabel('Instance ID')
    plt.ylabel(f'{metric_col.replace("runs_", "").capitalize()} Value')
    plt.title(f'{metric_col.replace("runs_", "").capitalize()} Performance by Instance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{chart_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_metric_chart(data, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics = ['runs_max', 'runs_min', 'runs_mean']
    titles = ['Max Performance', 'Min Performance', 'Mean Performance']

    color_generator = ColorGenerator()
    state_types = data['state_type'].unique()
    colors = color_generator.generate_colors(len(state_types))

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        for i, state_type in enumerate(state_types):
            subset = data[data['state_type'] == state_type]
            if len(subset) > 0:
                ax.plot(subset['instance_id'], subset[metric],
                        'o-', label=state_type, color=colors[i], linewidth=2, markersize=6)

        ax.set_xlabel('Instance ID')
        ax.set_ylabel('Performance Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/stage1_combined_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_analysis(data, dimension, save_dir):
    plt.figure(figsize=(12, 8))
    main_metric = f'instances_runs_{dimension}_mean'

    color_generator = ColorGenerator()
    state_types = data['state_type'].unique()
    colors = color_generator.generate_colors(len(state_types))

    values = []
    for state_type in state_types:
        subset = data[data['state_type'] == state_type]
        if len(subset) > 0:
            values.append(subset[main_metric].iloc[0])
        else:
            values.append(0)

    bars = plt.bar(state_types, values, color=colors)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.xlabel('State Type')
    plt.ylabel(f'{dimension.capitalize()} Performance')
    plt.title(f'Ablation Analysis - {dimension.capitalize()} Performance Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/stage2_ablation_single_{dimension}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_stage3_single_chart(data, dimension, save_dir, filters):
    plt.figure(figsize=(14, 8))
    metric_col = f'runs_{dimension}'

    if metric_col not in data.columns:
        print(f"警告: 列 {metric_col} 不存在于数据中")
        return

    color_generator = ColorGenerator()
    state_types = sorted(data['state_type'].unique())
    colors = color_generator.generate_colors(len(state_types))

    for i, state_type in enumerate(state_types):
        subset = data[data['state_type'] == state_type].copy()
        if len(subset) > 0:
            subset = subset.sort_values('instance_id')
            plt.plot(subset['instance_id'], subset[metric_col],
                     'o-', label=state_type, color=colors[i],
                     linewidth=2, markersize=6, alpha=0.8)

    plt.xlabel('Instance ID', fontsize=12)
    plt.ylabel(f'{dimension.capitalize()} Performance Value', fontsize=12)

    filter_str = " | ".join([f"{k}: {v}" for k, v in filters.items()]) if filters else "All Data"
    plt.title(f'Stage3 Analysis - {dimension.capitalize()} Performance by Instance\n({filter_str})',
              fontsize=14, pad=20)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filter_file_str = "_".join([f"{k}_{'-'.join(map(str, v))}" for k, v in filters.items()]) if filters else "all"
    filename = f'stage3_analysis_{dimension}_{filter_file_str}.png'

    plt.savefig(f'{save_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filename}")


def plot_stage3_combined_chart(data, save_dir, filters):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    dimensions = ['max', 'min', 'mean']
    titles = ['Max Performance', 'Min Performance', 'Mean Performance']

    color_generator = ColorGenerator()
    state_types = sorted(data['state_type'].unique())
    colors = color_generator.generate_colors(len(state_types))

    for idx, (dimension, title) in enumerate(zip(dimensions, titles)):
        ax = axes[idx]
        metric_col = f'runs_{dimension}'

        if metric_col not in data.columns:
            ax.text(0.5, 0.5, f'列 {metric_col} 不存在',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        for i, state_type in enumerate(state_types):
            subset = data[data['state_type'] == state_type].copy()
            if len(subset) > 0:
                subset = subset.sort_values('instance_id')
                ax.plot(subset['instance_id'], subset[metric_col],
                        'o-', label=state_type, color=colors[i],
                        linewidth=2, markersize=5, alpha=0.8)

        ax.set_xlabel('Instance ID', fontsize=11)
        ax.set_ylabel('Performance Value', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    filter_str = " | ".join([f"{k}: {v}" for k, v in filters.items()]) if filters else "All Data"
    fig.suptitle(f'Stage3 Combined Analysis - Performance by Instance ({filter_str})',
                 fontsize=16, y=1.02)

    plt.tight_layout()

    filter_file_str = "_".join([f"{k}_{'-'.join(map(str, v))}" for k, v in filters.items()]) if filters else "all"
    filename = f'stage3_combined_analysis_{filter_file_str}.png'

    plt.savefig(f'{save_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filename}")


def plot_stage1_charts(data, save_dir):
    """绘制阶段1图"""
    for metric_type in ['max', 'min', 'mean']:
        metric_col = f'runs_{metric_type}'
        plot_single_metric_chart(data, metric_col, f'stage1_{metric_type}_individual', save_dir)
    plot_combined_metric_chart(data, save_dir)


def plot_stage2_charts(stage2_data, save_dir):
    """绘制阶段2图 - 综合消融分析"""
    print("绘制阶段2图：综合消融分析...")
    
    # 为每个runs维度计算组件贡献度并绘图
    for runs_dimension in ['max', 'min', 'mean']:
        print(f"处理runs_{runs_dimension}维度...")
        
        # 计算组件贡献度和消融路径分析
        contributions, pathway_analysis, performance_better_when = calculate_component_contributions(stage2_data, runs_dimension)
        
        if len(contributions) == 0:
            print(f"runs_{runs_dimension}维度没有数据，跳过")
            continue
        

        # 按组合分组绘制
        grouped_data = contributions.groupby(['algorithm', 'city_num', 'mode', 'train_test'])
        plot_count = 0
        
        for group_name, group_data in grouped_data:
            if plot_count >= 3:  # 最多绘制3个组合
                break
                
            print(f"绘制组合: {group_name} (runs_{runs_dimension})")
            
            # 为每个组合创建一个大图 (3x2布局)
            fig, axes = plt.subplots(2, 2, figsize=(24, 18))
            fig.suptitle(f'{group_name[0]} | {group_name[1]} | {group_name[2]} | {group_name[3]} (runs_{runs_dimension})',
                         fontsize=16, fontweight='bold')
            
            try:
                # 为三个组件相关的图表生成统一的颜色方案
                base_states = ["current_city_onehot", "visited_mask", "order_embedding", "distances_from_current",'full']
                color_generator = ColorGenerator()
                unified_colors = color_generator.generate_colors(len(base_states), 'qualitative')
                color_map = dict(zip(base_states, unified_colors))
                
                # 绘制4个子图 - 重新排列顺序，将组件相关的三个图放到一起
                _plot_marginal_contributions_for_group(axes[0, 0], group_data, color_map)
                _plot_component_comparison_for_group(axes[1, 0], group_data, color_map)
                _plot_interaction_heatmap_for_group(axes[0, 1], group_data)
                _plot_degradation_from_pathway_data_for_group(axes[1, 1], group_data, pathway_analysis)
                
                # 设置子图标题
                axes[0, 0].set_title('Component Marginal Contributions', fontsize=10, fontweight='bold', pad=20)
                axes[1, 0].set_title('Component Comparison', fontsize=10, fontweight='bold', pad=20)
                axes[0, 1].set_title('Component Interaction Effects: AB - A - B', fontsize=10, fontweight='bold', pad=20)
                axes[1, 1].set_title('Performance Degradation Analysis', fontsize=10, fontweight='bold', pad=20)
                
                plt.tight_layout()
                plt.subplots_adjust(hspace=0.3, wspace=0.3)
                
                # 保存文件
                filename = f'stage2_comprehensive_ablation_analysis_{group_name[0]}_{group_name[1]}_{group_name[2]}_{group_name[3]}_runs_{runs_dimension}.png'
                plt.savefig(f'{save_dir}/{filename}', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"已保存: {filename}")
                plot_count += 1
                
            except Exception as e:
                print(f"绘制组合 {group_name} 时出现错误: {e}")
                plt.close()
                continue


def _plot_marginal_contributions_for_group(ax, group_data, color_map=None):
    """绘制边际贡献图"""
    try:
        if len(group_data) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 提取边际贡献数据
        marginal_cols = [col for col in group_data.columns if 'marginal_contribution' in col]
        if not marginal_cols:
            ax.text(0.5, 0.5, 'No marginal contribution data', ha='center', va='center', transform=ax.transAxes)
            return
        
        marginal_data = group_data[marginal_cols].mean()
        components = [col.replace('_marginal_contribution', '').replace('_', '\n') for col in marginal_cols]
        
        # 使用color_map通过key获取颜色
        if color_map is not None:
            colors = []
            for col in marginal_cols:
                component_key = col.replace('_marginal_contribution', '')
                colors.append(color_map.get(component_key, color_map.get('current_city_onehot', '#FF0000')))
        else:
            color_generator = ColorGenerator()
            colors = color_generator.generate_colors(len(components))
        
        bars = ax.bar(components, marginal_data.values, color=colors)
        
        # 添加数值标签
        for bar, value in zip(bars, marginal_data.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Performance Impact')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)


def _plot_interaction_heatmap_for_group(ax, group_data):
    """绘制交互效应热力图"""
    try:
        if len(group_data) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 提取交互效应数据
        interaction_cols = [col for col in group_data.columns if 'interaction' in col]
        if not interaction_cols:
            ax.text(0.5, 0.5, 'No interaction data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 构建交互矩阵
        components = ['current', 'visited', 'order', 'distances']
        n_components = len(components)
        interaction_matrix = np.zeros((n_components, n_components))
        
        interaction_data = group_data[interaction_cols].mean()
        
        for col, value in interaction_data.items():
            parts = col.replace('_interaction', '').split('_')
            if len(parts) >= 2:
                comp1, comp2 = parts[0], parts[1]
                try:
                    idx1 = components.index(comp1)
                    idx2 = components.index(comp2)
                    interaction_matrix[idx1, idx2] = value
                    interaction_matrix[idx2, idx1] = value
                except ValueError:
                    continue
        
        # 绘制热力图
        sns.heatmap(interaction_matrix, annot=True, fmt='.3f',
                    xticklabels=[c.capitalize() for c in components],
                    yticklabels=[c.capitalize() for c in components],
                    cmap='RdBu_r', center=0, ax=ax)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)


def _plot_significance_tests_for_group(ax, group_data):
    """绘制统计显著性检验结果"""
    try:
        if len(group_data) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 提取显著性检验数据
        p_value_cols = [col for col in group_data.columns if 'p_value' in col]
        effect_size_cols = [col for col in group_data.columns if 'effect_size' in col]
        
        if not p_value_cols or not effect_size_cols:
            ax.text(0.5, 0.5, 'No significance test data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        p_values = group_data[p_value_cols].mean()
        effect_sizes = group_data[effect_size_cols].mean()
        
        # 提取组件名称
        component_names = []
        for col in p_value_cols:
            part = col.split('remove_')[1].split("_p_value")[0]
            component_names.append(part)

        
        # 直接使用原始p值，不进行log变换
        p_values_raw = p_values.values
        
        # 为不同组件生成不同颜色
        color_generator = ColorGenerator()
        unique_components = list(set(component_names))
        component_colors = color_generator.generate_colors(len(unique_components), 'qualitative')
        component_color_map = dict(zip(unique_components, component_colors))
        
        # 为每个组件分配颜色
        colors = [component_color_map[name] for name in component_names]
        
        # 绘制散点图 - 为每个组件单独绘制以便添加到图例
        from matplotlib.patches import Patch
        legend_elements = []
        
        for i, (component, color) in enumerate(zip(component_names, colors)):
            ax.scatter(effect_sizes.values[i], p_values_raw[i], c=color, s=100, alpha=0.7, edgecolors='black')
            
            # 只为每个唯一组件添加一次图例项
            if component not in [elem.get_label() for elem in legend_elements]:
                legend_elements.append(Patch(facecolor=color, alpha=0.7, label=component.title()))
        
        # 添加显著性阈值线
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05 threshold')
        ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='p=0.1 threshold')
        
        # 添加阈值线到图例
        from matplotlib.lines import Line2D
        legend_elements.extend([
            Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='p=0.05 threshold'),
            Line2D([0], [0], color='orange', linestyle='--', alpha=0.5, label='p=0.1 threshold')
        ])
        
        # 添加组件标签
        for i, name in enumerate(component_names):
            if i < len(effect_sizes.values) and i < len(p_values_raw):
                ax.annotate(name.title(), (effect_sizes.values[i], p_values_raw[i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=8, ha='left')
        
        ax.set_xlabel('Effect Size (Cohen\'s d)')
        ax.set_ylabel('p-value')
        ax.legend(handles=legend_elements, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.0)  # p值范围在0-1之间
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)


def _plot_importance_ranking_for_group(ax, group_data, color_map=None):
    """绘制重要性排序"""
    try:
        if len(group_data) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        impact_cols = [col for col in group_data.columns if 'impact_magnitude' in col]
        
        if not impact_cols:
            # 使用marginal_contribution的绝对值
            marginal_cols = [col for col in group_data.columns if 'marginal_contribution' in col]
            if marginal_cols:
                marginal_data = group_data[marginal_cols].mean().abs()
                components = [col.replace('_marginal_contribution', '').replace('_', '\n').title()
                              for col in marginal_cols]
                importance_scores = marginal_data.values
                component_keys = [col.replace('_marginal_contribution', '') for col in marginal_cols]
            else:
                ax.text(0.5, 0.5, 'No importance data available', ha='center', va='center', transform=ax.transAxes)
                return
        else:
            impact_data = group_data[impact_cols].mean()
            components = [col.replace('_impact_magnitude', '').replace('_', '\n').title()
                          for col in impact_cols]
            importance_scores = impact_data.values
            component_keys = [col.replace('_impact_magnitude', '') for col in impact_cols]
        
        # 按重要性排序
        sorted_indices = np.argsort(importance_scores)[::-1] #倒序
        components = [components[i] for i in sorted_indices]
        component_keys = [component_keys[i] for i in sorted_indices]
        importance_scores = importance_scores[sorted_indices]
        
        # 归一化
        max_score = max(importance_scores) if max(importance_scores) > 0 else 1
        normalized_scores = importance_scores / max_score
        
        # 使用color_map通过key获取颜色
        if color_map is not None:
            colors = []
            for key in component_keys:
                colors.append(color_map.get(key, color_map.get('current_city_onehot', '#FF0000')))
        else:
            color_generator = ColorGenerator()
            colors = color_generator.generate_colors(len(components))
        
        # 绘制水平条形图
        bars = ax.barh(components, normalized_scores, color=colors)
        
        # 添加数值标签
        for bar, value in zip(bars, normalized_scores):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2.,
                    f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Normalized Importance Score')
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)


def _plot_component_comparison_for_group(ax, group_data, color_map=None):
    """绘制组件对比图 - 直接使用performance_dict的值"""
    try:
        if len(group_data) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return

        # 从透传的performance_dict中获取所有状态类型的性能值
        row = group_data.iloc[0]
        
        # 获取full状态的性能
        full_performance = row.get('full', 0)
        
        # 基础组件列表
        base_components = ["current_city_onehot", "visited_mask", "order_embedding", "distances_from_current"]
        component_labels = ["Current City", "Visited Mask", "Order Embedding", "Distances"]
        
        # 提取单个组件移除后的性能值
        single_removal_performance = []
        available_components = []
        available_labels = []
        
        for component, label in zip(base_components, component_labels):
            # 构建移除该组件后的状态键名
            component_short = component.split('_')[0]  # current, visited, order, distances
            removal_key = f'ablation_remove_{component_short}'
            
            if removal_key in row and pd.notna(row[removal_key]):
                single_removal_performance.append(row[removal_key])
                available_components.append(component)
                available_labels.append(label)
        
        if not available_components:
            ax.text(0.5, 0.5, 'No component removal data available', ha='center', va='center', transform=ax.transAxes)
            return

        x_pos = np.arange(len(available_components))

        # 使用color_map获取颜色
        if color_map is not None:
            colors = [color_map.get(comp, '#FF0000') for comp in available_components]
        else:
            color_generator = ColorGenerator()
            colors = color_generator.generate_colors(len(available_components))

        # 绘制对比条形图
        bars1 = ax.bar(x_pos - 0.2, [full_performance] * len(available_components), 0.4,
                       label='Full State', color=color_map.get('full', 'green') if color_map else 'green', alpha=0.7)
        bars2 = ax.bar(x_pos + 0.2, single_removal_performance, 0.4,
                       label='Component Removed', color=colors, alpha=0.7)

        # 添加数值标签
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.01,
                    f'{full_performance:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.01,
                    f'{single_removal_performance[i]:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Components')
        ax.set_ylabel('Performance')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(available_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)


def _plot_performance_summary_for_group(ax, group_data):
    """绘制性能总结"""
    try:
        if len(group_data) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 提取基本信息
        full_performance = group_data['full_performance'].iloc[0] if 'full_performance' in group_data.columns else 0
        runs_dimension = group_data['runs_dimension'].iloc[0] if 'runs_dimension' in group_data.columns else 'unknown'
        
        # 提取边际贡献数据
        marginal_cols = [col for col in group_data.columns if 'marginal_contribution' in col]
        marginal_data = group_data[marginal_cols].mean() if marginal_cols else pd.Series()
        
        # 计算统计信息
        total_impact = marginal_data.abs().sum() if len(marginal_data) > 0 else 0
        max_impact = marginal_data.abs().max() if len(marginal_data) > 0 else 0
        min_impact = marginal_data.abs().min() if len(marginal_data) > 0 else 0
        
        # 创建汇总文本
        summary_text = f"""Performance Summary (runs_{runs_dimension})

Full State Performance: {full_performance:.3f}

Component Impact Analysis:
• Total Impact: {total_impact:.3f}
• Max Impact: {max_impact:.3f}
• Min Impact: {min_impact:.3f}
• Components Analyzed: {len(marginal_data)}

Key Insights:
• Most Critical Component: {marginal_data.abs().idxmax().replace('_marginal_contribution', '') if len(marginal_data) > 0 else 'N/A'}
• Least Critical Component: {marginal_data.abs().idxmin().replace('_marginal_contribution', '') if len(marginal_data) > 0 else 'N/A'}
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)


def plot_stage3_charts(df_original, target_metric, save_dir, filters=None):
    """绘制阶段3图 - 使用test数据重新生成阶段1数据"""
    print("为阶段3图重新生成基于test数据的阶段1数据...")

    # 使用封装的函数重新生成测试数据
    stage0_test_data = process_stage0_data(df_original, target_metric, 'test')
    stage1_test_data = process_stage1_data(stage0_test_data)
    stage0_test_data.to_csv("stage0_test_data.csv", index=False)
    stage1_test_data.to_csv("stage1_test_data.csv", index=False)
    # stage0_test_data = pd.read_csv("stage0_test_data.csv")
    # stage1_test_data = pd.read_csv("stage1_test_data.csv")

    # 使用测试数据进行绘图
    data = stage1_test_data.copy()

    # 应用筛选条件
    if filters:
        print(f"应用筛选条件: {filters}")
        original_shape = data.shape[0]
        for key, values in filters.items():
            if key in data.columns:
                data = data[data[key].isin(values)]
        print(f"筛选前后数据变化: {original_shape} -> {data.shape[0]}")

    if data.empty:
        print("警告: 筛选后数据为空，跳过绘图")
        return

    for dimension in ['max', 'min', 'mean']:
        plot_stage3_single_chart(data, dimension, save_dir, filters)
    plot_stage3_combined_chart(data, save_dir, filters)

def process_single_data_file(df: pd.DataFrame, index: int, target_metric: str, base_save_dir: str):
    """
    处理单个数据

    Args:
        df: 输入的pandas DataFrame
        index: 文件索引
        target_metric: 目标指标
        base_save_dir: 基础保存目录
    """
    print(f"\n{'=' * 80}")
    print(f"处理数据 {index}")
    print(f"{'=' * 80}")

    # 创建该数据文件对应的文件夹
    current_save_dir = os.path.join(base_save_dir, f"data_{index}")
    os.makedirs(current_save_dir, exist_ok=True)

    try:
        # 数据处理流程 - 使用封装的函数
        print("开始数据处理流程")
        print("-" * 40)

        # 使用封装的函数处理各阶段数据
        stage0_data = process_stage0_data(df, target_metric, 'train')
        stage0_data.to_csv("stage0_data.csv",index=False)
        stage1_data = process_stage1_data(stage0_data)
        stage1_data.to_csv("stage1_data.csv",index=False)
        # stage0_data = pd.read_csv("stage0_data.csv")
        # stage1_data = pd.read_csv("stage1_data.csv")


        stage2_data = process_stage2_data(stage1_data)

        print("开始实验1：不同状态组合的消融实验")
        print("-" * 40)

        # 实验1：消融实验分析
        print("绘制阶段1图：instance_id维度分析")
        plot_stage1_charts(stage1_data, current_save_dir)

        print("绘制阶段2图：综合消融分析")
        plot_stage2_charts(stage2_data, current_save_dir)

        print("开始实验2：基于阶段1数据的详细分析")
        print("-" * 40)

        # 实验2：基于阶段1数据的分析（使用test数据）
        print("绘制阶段3图：基于test数据的详细分析")

        # 示例筛选条件 - 可以根据实际数据调整
        filters_examples = [
            None,  # 不筛选，显示所有数据
            # {'mode': ['per_instance'],
            #  'state_type': ['full']},  # 组合筛选
        ]

        for i, filters in enumerate(filters_examples):
            print(f"  绘制筛选条件 {i + 1}: {filters}")
            plot_stage3_charts(df, target_metric, current_save_dir, filters)

        print(f"数据 {index} 分析完成！")
        print(f"图表已保存到: {current_save_dir}")

    except Exception as e:
        print(f"处理数据 {index} 时出错: {e}")
        print(f"详细错误信息: {traceback.format_exc()}")


def main():
    """主函数"""
    # 配置参数 - 支持多个数据路径
    data_paths = [
        "/home/y/workplace/code/git_code/clean/RL-TSP-Solver/results/tsp_rl_ablation_DQN_per_instance_20250813_115704/experiment_data_20250813_115704.csv",  # 请替换为实际数据路径
        # "/home/y/workplace/mac-bk/git_code/clean/dissertation/results/tsp_rl_ablation_DQN_cross_instance_20250803_233953/experiment_data_20250803_233953.csv"
    ]
    target_metric = "optimality_gap"  # 对比对象：optimality_gap 或 total_reward
    base_save_dir = "./experiment_plots"
    os.makedirs(base_save_dir, exist_ok=True)

    print(f"开始处理 {len(data_paths)} 个数据文件")
    print(f"目标指标: {target_metric}")
    print(f"基础保存目录: {base_save_dir}")

    # 处理每个数据文件
    for index, data_path in enumerate(data_paths):
        # index = 1
        if os.path.exists(data_path):
            print(f"加载数据文件: {data_path}")
            columns = [
                'algorithm', 'city_num', 'mode', 'instance_id', 'run_id', 'state_type',
                'train_test', 'episode', 'step',
                'state', 'done', 'reward',
                'total_reward', 'current_distance', 'optimal_distance',
                'state_values'
            ]
            dtype_dict = {'instance_id': str}
            
            df = pd.read_csv(data_path, usecols=columns, dtype=dtype_dict)
            print(f"数据加载完成，形状: {df.shape}")
            # df = None
            process_single_data_file(df, index, target_metric, base_save_dir)
        else:
            print(f"警告: 数据文件 {data_path} 不存在，跳过处理")

    print(f"结果保存在以下目录中:")
    for index in range(len(data_paths)):
        result_dir = os.path.join(base_save_dir, f"data_{index}")
        if os.path.exists(result_dir):
            print(f"  - data_{index}: {result_dir}")


if __name__ == "__main__":
    main()
