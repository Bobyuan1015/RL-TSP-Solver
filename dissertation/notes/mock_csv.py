import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple, Any
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error
import os
from itertools import combinations

warnings.filterwarnings('ignore')


# 构建状态变量映射关系
def build_state_combinations():
    """构建消融实验的状态组合映射"""
    base_states = ["current_city_onehot", "visited_mask", "order_embedding", "distances_from_current"]

    map_state_types = {}

    # 1. 全状态（基线）
    map_state_types['full'] = base_states.copy()

    # 2. 依次移除一种状态（4种组合）
    for i, state_to_remove in enumerate(base_states):
        remaining_states = [s for s in base_states if s != state_to_remove]
        map_state_types[f'ablation_remove_{state_to_remove.split("_")[0]}'] = remaining_states

    # 3. 依次移除两种状态（6种组合）
    for i, (state1, state2) in enumerate(combinations(base_states, 2)):
        remaining_states = [s for s in base_states if s not in [state1, state2]]
        key_name = f'ablation_remove_{state1.split("_")[0]}_{state2.split("_")[0]}'
        map_state_types[key_name] = remaining_states

    return map_state_types


# 全局映射关系
map_state_types = build_state_combinations()


class TSPAblationExperimentGenerator:
    """TSP消融实验数据生成器"""

    def __init__(self):
        self.algorithms = ['DQN_visited', 'DQN_LSTM', 'Reinforce', 'ActorCritic', 'DQN_order', 'DQN_optimal']
        self.state_types = list(map_state_types.keys())
        self.city_nums = [10, 20, 30, 50]
        self.modes = ['per_instance', 'cross_instance']
        self.train_test_splits = ['train', 'test']
        self.total_instances = 100
        self.train_instances = 80
        self.test_instances = 20
        self.runs_per_instance = 5

        # 设置随机种子以保证可重复性
        np.random.seed(42)

    def generate_optimal_distances(self, city_num: int, instance_id: int) -> float:
        """生成TSP实例的最优距离（基于城市数量的经验公式）"""
        np.random.seed(instance_id)  # 确保每个实例的最优解固定
        base_distance = np.sqrt(city_num) * 10
        variation = np.random.normal(0, 0.1) * base_distance
        return max(base_distance + variation, base_distance * 0.8)

    def generate_state_representation(self, state_type: str, city_num: int, step: int) -> str:
        """根据状态类型生成对应的状态表示"""
        current_city = np.random.randint(0, city_num)
        visited_mask = np.random.choice([0, 1], size=city_num, p=[0.3, 0.7])

        # 获取该状态类型应包含的组件
        state_components = map_state_types[state_type]

        state = {}

        # 根据状态组件列表构建状态
        if "current_city_onehot" in state_components:
            state['current_city_onehot'] = [1 if i == current_city else 0 for i in range(city_num)]

        if "visited_mask" in state_components:
            state['visited_mask'] = visited_mask.tolist()

        if "order_embedding" in state_components:
            state['order_embedding'] = [step / 100.0] * city_num

        if "distances_from_current" in state_components:
            state['distances_from_current'] = np.random.exponential(2, city_num).tolist()

        return json.dumps(state)

    def calculate_state_performance_impact(self, state_type: str, algorithm: str, city_num: int) -> Dict[str, float]:
        """根据消融实验理论计算状态类型对性能的影响"""
        # 基础性能参数
        algorithm_performance = {
            'DQN_visited': 0.75, 'DQN_LSTM': 0.80, 'Reinforce': 0.70,
            'ActorCritic': 0.85, 'DQN_order': 0.78, 'DQN_optimal': 0.90
        }

        # 组件重要性权重（基于TSP领域知识）
        component_importance = {
            'current_city_onehot': 0.30,  # 当前位置信息，基础且关键
            'visited_mask': 0.35,  # 访问状态，避免重复访问
            'order_embedding': 0.20,  # 访问顺序，影响路径优化
            'distances_from_current': 0.15  # 距离信息，局部优化辅助
        }

        # 计算状态组合的总重要性
        state_components = map_state_types[state_type]
        total_importance = sum(component_importance[comp] for comp in state_components)

        # 规模惩罚
        scale_penalty = 1 - (city_num - 10) * 0.015
        base_performance = algorithm_performance[algorithm] * scale_penalty

        # 根据组件重要性调整性能
        performance_multiplier = total_importance
        performance = base_performance * performance_multiplier

        return {
            'base_optimality_gap': (1 - performance) * 100,
            'convergence_factor': performance_multiplier,
            'stability_factor': 1 / max(performance_multiplier, 0.1),
            'component_importance': total_importance
        }

    def generate_episode_data(self, algorithm: str, city_num: int, mode: str,
                              state_type: str, instance_id: int, run_id: int,
                              train_test: str) -> List[Dict]:
        """生成单个实例的episode数据"""
        data = []
        optimal_distance = self.generate_optimal_distances(city_num, instance_id)
        optimal_path = f"optimal_path_{instance_id}"

        # 根据状态类型和算法设定episode数量
        max_episodes = 100 if train_test == 'train' else 50
        state_value = map_state_types[state_type]
        performance_params = self.calculate_state_performance_impact(state_type, algorithm, city_num)

        np.random.seed(instance_id * 1000 + run_id * 100)

        for episode in range(max_episodes):
            episode_length = city_num
            episode_reward = 0
            current_distance = 0

            for step in range(episode_length):
                # 生成状态表示
                state = self.generate_state_representation(state_type, city_num, step)

                # 模拟奖励和损失
                step_distance = np.random.exponential(2)
                current_distance += step_distance

                # 奖励计算
                step_reward = -step_distance
                if step == episode_length - 1:
                    step_reward -= np.random.exponential(3)

                episode_reward += step_reward

                # 损失计算
                loss = np.random.exponential(0.5) if algorithm.startswith('DQN') else np.nan

                # 根据性能参数调整最终距离
                optimality_gap = performance_params['base_optimality_gap']
                final_distance = optimal_distance * (1 + optimality_gap / 100)

                data.append({
                    'algorithm': algorithm,
                    'city_num': city_num,
                    'mode': mode,
                    'instance_id': instance_id,
                    'run_id': run_id,
                    'state_type': state_type,
                    'train_test': train_test,
                    'episode': episode,
                    'step': step,
                    'state': state,
                    'done': 1 if step == episode_length - 1 else 0,
                    'reward': step_reward,
                    'loss': loss,
                    'total_reward': episode_reward if step == episode_length - 1 else np.nan,
                    'current_distance': final_distance if step == episode_length - 1 else np.nan,
                    'optimal_distance': optimal_distance,
                    'optimal_path': optimal_path,
                    'coordinates': [],
                    'state_values': state_value
                })

        return data

    def generate_full_experiment_data(self) -> pd.DataFrame:
        """生成完整的实验数据"""
        all_data = []

        # 选择实验参数
        selected_algorithms = ['DQN_LSTM', 'ActorCritic'][:1]
        selected_cities = [10, 20][:1]
        selected_instances = list(range(0, 10, 2))

        total_combinations = len(selected_algorithms) * len(selected_cities) * len(self.modes) * len(self.state_types)
        current_combination = 0

        print("开始生成消融实验数据...")
        print(f"状态组合总数: {len(self.state_types)}")

        for algorithm in selected_algorithms:
            for city_num in selected_cities:
                for mode in self.modes:
                    for state_type in self.state_types:
                        current_combination += 1
                        print(
                            f"处理组合 {current_combination}/{total_combinations}: {algorithm}-{city_num}-{mode}-{state_type}")

                        if mode == 'per_instance':
                            # 训练集
                            for instance_id in selected_instances[:2]:
                                for run_id in range(2):
                                    episode_data = self.generate_episode_data(
                                        algorithm, city_num, mode, state_type,
                                        instance_id, run_id, 'train'
                                    )
                                    all_data.extend(episode_data[-20:])

                            # 测试集
                            for instance_id in selected_instances[2:3]:
                                for run_id in range(2):
                                    episode_data = self.generate_episode_data(
                                        algorithm, city_num, mode, state_type,
                                        instance_id, run_id, 'test'
                                    )
                                    all_data.extend(episode_data[-10:])

                        else:  # cross_instance
                            for run_id in range(2):
                                for instance_id in selected_instances[:2]:
                                    episode_data = self.generate_episode_data(
                                        algorithm, city_num, mode, state_type,
                                        instance_id, run_id, 'train'
                                    )
                                    all_data.extend(episode_data[-15:])

                                for instance_id in selected_instances[2:3]:
                                    episode_data = self.generate_episode_data(
                                        algorithm, city_num, mode, state_type,
                                        instance_id, run_id, 'test'
                                    )
                                    all_data.extend(episode_data[-8:])

        df = pd.DataFrame(all_data)
        print(f"数据生成完成！总共 {len(df)} 行数据")
        return df




try:
    # 1. 显示状态组合映射
    print("=" * 80)
    print("TSP消融实验状态组合映射")
    print("=" * 80)
    for state_type, components in map_state_types.items():
        print(f"{state_type}: {components}")
    print("=" * 80)

    # 2. 创建实验数据生成器
    print("\n步骤 1: 初始化实验数据生成器...")
    generator = TSPAblationExperimentGenerator()

    # 3. 生成实验数据
    print("\n步骤 2: 生成实验数据...")
    df = generator.generate_full_experiment_data()

    # 3. 保存原始数据
    print("\n步骤 3: 保存原始数据...")
    df.to_csv('tsp_ablation_experiment_data.csv', index=False)

    print("📁 生成的文件包括:")
    print("├─ tsp_ablation_experiment_data.csv (原始数据)")


except Exception as e:
    print(f"详细错误信息: {traceback.format_exc()}")

