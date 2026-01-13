# run_visualization.py
"""
双层知识图谱可视化生成脚本 (Final Version)
功能：
1. 加载数据和KG模块
2. 自动筛选包含有效病史(Layer 2)和视觉特征(Layer 1)的病例
3. 生成可视化图片到 'final_visualizations' 目录
4. 生成满 5 组后自动停止
"""
import sys
import os
import pandas as pd
import json
import numpy as np

# ---------------------------------------------------------
# 1. 配置区域 (请确保路径与您环境一致)
# ---------------------------------------------------------
DATA_DIR = 'data'
# 指向旧版融合向量 (用于检索)
VISUAL_DIR = 'visual/visual_features/multi_scale' 

RADGRAPH_JSON = os.path.join(DATA_DIR, 'graph/radgraph/MIMIC-CXR_graphs.json')
CSV_PATH = os.path.join(DATA_DIR, 'processed_dataset.csv')
KB_PATH = os.path.join(DATA_DIR, 'disease_knowledge_base.json')

# 图片保存目录
OUTPUT_DIR = 'final_visualizations' 
# 目标生成数量
TARGET_COUNT = 5  

# ---------------------------------------------------------
# 2. 导入模块
# ---------------------------------------------------------
try:
    # 1. 基础加载器
    from kg_module.radgraph_loader import RadGraphLoader, CaseDatabase
    # 2. 从 dynamic_kg_module.py 导入 Layer 1 构建函数
    from kg_module.dynamic_kg_module import build_kg_module
    # 3. 从 disease_kg_module.py 导入 Layer 2 构建函数
    from kg_module.disease_kg_module import build_disease_kg_module
    # 4. 可视化工具
    from kg_module.kg_visualizer import KGVisualizer
    # 5. 病史解析器
    from kg_module.disease_graph_builder import PatientHistoryParser
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("确认你在 ~/TPY/kg_contrast 目录下运行，并且文件路径正确。")
    sys.exit(1)

# ---------------------------------------------------------
# 3. 辅助函数
# ---------------------------------------------------------
def safe_parse_history(h):
    """安全解析 history 字符串"""
    if pd.isna(h) or h == '' or h == '[]':
        return []
    try: 
        return json.loads(h) if isinstance(h, str) else []
    except: 
        try:
            return eval(h.replace('null', 'None')) if isinstance(h, str) else []
        except:
            return []

# ---------------------------------------------------------
# 4. 主逻辑
# ---------------------------------------------------------
def main():
    print(f"=== 启动可视化生成器 (目标: {TARGET_COUNT} 组) ===")
    
    # 1. 准备输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 创建目录: {OUTPUT_DIR}")
    
    visualizer = KGVisualizer(save_dir=OUTPUT_DIR)
    
    # 2. 加载模块
    print("\n[1/3] 正在加载数据和模型...")
    
    if not os.path.exists(VISUAL_DIR):
        print(f"❌ 错误: 视觉特征目录不存在: {VISUAL_DIR}")
        return

    rad_loader = RadGraphLoader(RADGRAPH_JSON)
    
    # 初始化数据库 (指定正确的 visual_feature_dir)
    case_db = CaseDatabase(CSV_PATH, visual_feature_dir=VISUAL_DIR, radgraph_loader=rad_loader)
    
    if len(case_db.database) == 0:
        print("❌ 错误: CaseDatabase 未加载到任何数据，请检查 VISUAL_DIR 路径是否正确。")
        return

    # 构建 KG 模块
    kg1_module = build_kg_module(case_db.database, config={'retriever': {'top_k': 3}})
    kg2_module = build_disease_kg_module(KB_PATH)
    
    print("✅ 模块加载完成。")
    
    # 3. 开始筛选和生成
    print("\n[2/3] 开始筛选病例并绘图...")
    df = pd.read_csv(CSV_PATH)
    
    success_count = 0
    
    # 遍历数据集
    for idx, row in df.iterrows():
        # 达到目标数量则停止
        if success_count >= TARGET_COUNT:
            print(f"\n✨ 已达到目标数量 ({TARGET_COUNT})，任务完成。")
            break
            
        study_id = row['study_id']
        
        # --- 筛选条件 1: 必须有 Layer 1 数据 (视觉特征) ---
        if study_id not in case_db.database:
            continue
            
        # --- 筛选条件 2: 必须有 Layer 2 数据 (非空病史) ---
        history = safe_parse_history(row['history'])
        if not history or len(history) == 0:
            continue

        try:
            # === 处理 Layer 2 (病史图) ===
            # 实例化解析器
            parser = PatientHistoryParser(current_study_datetime=row['study_datetime'])
            report_text = row['report'] if pd.notna(row['report']) else None
            parsed_hist = parser.parse(history, [report_text] if report_text else None)
            
            # 构建图
            layer2_graph = kg2_module.graph_builder.build_patient_graph(parsed_hist, max_entities=15)
            
            # 如果图是空的（例如病史里的ICD在知识库里找不到），则跳过
            if not layer2_graph.get('entities'):
                continue

            # === 处理 Layer 1 (视觉检索图) ===
            visual_feat = case_db.database[study_id]['visual_feat']
            retrieval_results = kg1_module.retriever.retrieve(visual_feat, exclude_study_id=study_id)
            
            if not retrieval_results:
                continue
                
            # 聚合
            cases = kg1_module.retriever.get_retrieved_cases(retrieval_results)
            radgraphs = [c['radgraph'] for c in cases]
            sims = [c['similarity'] for c in cases]
            layer1_graph = kg1_module.aggregator.aggregate(radgraphs, sims)

            # === 保存图片 ===
            print(f"  🎨 正在生成 Case {study_id} ... ({success_count + 1}/{TARGET_COUNT})")
            
            # 调用可视化器
            visualizer.visualize_layer1(layer1_graph, str(study_id))
            visualizer.visualize_layer2(layer2_graph, str(study_id))
            
            success_count += 1
            
        except Exception as e:
            # 遇到个别坏数据不中断，打印错误并继续
            print(f"  ⚠️ 跳过 Case {study_id}: {str(e)}")
            continue

    # 4. 结束
    if success_count == 0:
        print("\n❌ 未生成任何图片。可能原因：没有同时满足有病史且有特征的病例。")
    else:
        print(f"\n[3/3] 全部完成！请查看文件夹: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()