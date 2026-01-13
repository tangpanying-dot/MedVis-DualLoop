# kg_module/kg_visualizer.py
import networkx as nx
import matplotlib.pyplot as plt
import os
from typing import Dict, List

class KGVisualizer:
    def __init__(self, save_dir='visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _filter_graph(self, entities: Dict, relations: List, max_nodes: int = 20):
        """
        内部辅助函数：强制筛选 Top-K 节点，防止画成毛线球
        """
        print(f"  [_filter_graph] 输入: {len(entities)} 实体, {len(relations)} 关系")
        
        if len(entities) <= max_nodes:
            print(f"  [_filter_graph] 实体数 <= {max_nodes}，不过滤")
            return entities, relations

        # 1. 按权重排序，取 Top-K
        sorted_entities = sorted(
            entities.items(), 
            key=lambda x: x[1].get('weight', 0), 
            reverse=True
        )
        top_entities = dict(sorted_entities[:max_nodes])
        top_keys = set(top_entities.keys())

        print(f"  [_filter_graph] Top {max_nodes} 实体: {list(top_keys)[:5]}...")

        # 2. 过滤关系：只保留两端都在 Top-K 里的边
        filtered_relations = []
        for src, tgt, rel, w in relations:
            if src in top_keys and tgt in top_keys:
                filtered_relations.append((src, tgt, rel, w))
            else:
                print(f"  [_filter_graph] 过滤掉边: {src} -> {tgt} (src_in={src in top_keys}, tgt_in={tgt in top_keys})")
        
        print(f"  [_filter_graph] 输出: {len(top_entities)} 实体, {len(filtered_relations)} 关系")
        return top_entities, filtered_relations

    def visualize_layer1(self, graph_data: Dict, study_id: str):
        """可视化 Layer 1 (RadGraph)"""
        raw_entities = graph_data.get('entities', {})
        raw_relations = graph_data.get('relations', [])
        
        if not raw_entities:
            return

        # === 强制过滤 ===
        entities, relations = self._filter_graph(raw_entities, raw_relations, max_nodes=20)
        
        G = nx.DiGraph()
        
        # 1. 添加节点
        node_colors = []
        for label, data in entities.items():
            G.add_node(label)
            etype = data.get('type', '')
            if 'ANAT' in etype:
                node_colors.append('#AED6F1') # 蓝 (解剖)
            elif 'OBS' in etype:
                node_colors.append('#F5B7B1') # 红 (发现)
            else:
                node_colors.append('#D2B4DE') # 紫

        # 2. 添加边
        edge_labels = {}
        for src, tgt, rel_type, weight in relations:
            G.add_edge(src, tgt)
            short_rel = rel_type.replace('located_at', 'loc').replace('modify', 'mod')
            edge_labels[(src, tgt)] = short_rel

        # 3. 绘图
        self._plot_graph(G, node_colors, edge_labels, 
                        title=f"Layer 1: Visual Context (Study {study_id})",
                        filename=f"{study_id}_layer1.png")

    def visualize_layer2(self, graph_data: Dict, study_id: str):
        """可视化 Layer 2 (Disease KG)"""
        raw_entities = graph_data.get('entities', {})
        raw_relations = graph_data.get('relations', [])
        
        print(f"\n[可视化调试] Study {study_id}")
        print(f"  原始数据: {len(raw_entities)} 实体, {len(raw_relations)} 关系")
        
        if not raw_entities:
            print(f"  ⚠️ 实体为空，跳过")
            return

        # === 强制过滤 ===
        entities, relations = self._filter_graph(raw_entities, raw_relations, max_nodes=20)
        
        print(f"  过滤后: {len(entities)} 实体, {len(relations)} 关系")
        if len(relations) > 0:
            print(f"  示例关系: {relations[0]}")
        else:
            print(f"  ⚠️ 警告: 过滤后没有关系！")
            if raw_relations:
                print(f"  原始关系前3个: {raw_relations[:3]}")

        G = nx.DiGraph()
        node_colors = []
        node_sizes = []
        
        # 确定核心阈值
        weights = [d.get('weight', 0) for d in entities.values()]
        max_w = max(weights) if weights else 1.0

        # ===== 建立 ICD -> Name 的映射 =====
        icd_to_name = {}
        for icd, data in entities.items():
            name = data.get('name', icd)
            # 截断过长的名字
            if len(name) > 20: 
                name = name[:18] + ".."
            icd_to_name[icd] = name
            
            G.add_node(name)
            
            weight = data.get('weight', 0)
            if weight >= max_w * 0.8 and weight > 0:
                node_colors.append('#F8C471')  # 橙 (高风险)
                node_sizes.append(2500)
            else:
                node_colors.append('#A9DFBF')  # 绿 (背景)
                node_sizes.append(1500)

        # ===== 使用映射来添加边 =====
        edge_labels = {}
        edges_added = 0
        for src_icd, tgt_icd, rel_type, _ in relations:
            # 使用 ICD 去查找对应的 name
            src_name = icd_to_name.get(src_icd)
            tgt_name = icd_to_name.get(tgt_icd)
            
            if src_name is None:
                print(f"  ⚠️ 源节点 {src_icd} 不在映射中")
                continue
            if tgt_name is None:
                print(f"  ⚠️ 目标节点 {tgt_icd} 不在映射中")
                continue
            
            # 只有当两个节点都在图中时才添加边
            if src_name in G.nodes and tgt_name in G.nodes:
                G.add_edge(src_name, tgt_name)
                # 简化关系名
                if 'evolved' in rel_type:
                    short_rel = '→'
                elif 'co-occurring' in rel_type:
                    short_rel = '↔'
                else:
                    short_rel = rel_type.replace('progresses_to', '→').replace('causes', '⇒')
                edge_labels[(src_name, tgt_name)] = short_rel
                edges_added += 1
            else:
                print(f"  ⚠️ 节点不在图中: src={src_name in G.nodes}, tgt={tgt_name in G.nodes}")
        
        print(f"  最终添加到图中的边数: {edges_added}")
        print(f"  NetworkX图统计: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

        self._plot_graph(G, node_colors, edge_labels, 
                        title=f"Layer 2: Disease Risk (Study {study_id})",
                        filename=f"{study_id}_layer2.png",
                    node_sizes=node_sizes)
    def _plot_graph(self, G, node_colors, edge_labels, title, filename, node_sizes=2000):
        """通用绘图函数"""
        plt.figure(figsize=(12, 9))  # 稍微增大画布
        try:
            # ===== 只改这里：增大 k 值让节点更分散 =====
            pos = nx.spring_layout(G, k=3.0, iterations=150, seed=42)
            
            # 确保 node_sizes 格式正确
            if isinstance(node_sizes, list):
                if len(node_sizes) != len(G.nodes):
                    node_sizes = 2000
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                                edgecolors='gray', alpha=0.95)
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                                arrowsize=15, width=1.2, alpha=0.6)
            
            # 字体设置（保持原样）
            nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif', 
                                font_weight='bold')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                        font_size=7, font_color='#333333')
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.axis('off')
            
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  -> Saved visualization: {filename}")
            
        except Exception as e:
            print(f"  Visualizer Error: {e}")
        finally:
            plt.close()