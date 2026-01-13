import pandas as pd
import os

# ================= 配置路径 =================
INPUT_CSV = 'data/processed_dataset.csv'  # 你的全量数据 (v17)
OFFICIAL_SPLIT = 'data/mimic-cxr/mimic-cxr-2.0.0-split.csv'

# 输出路径
OUTPUT_TRAIN = 'data/processed_dataset_train.csv'
OUTPUT_VAL   = 'data/processed_dataset_val.csv'
OUTPUT_TEST  = 'data/processed_dataset_test.csv'

def split_by_official_list_v2():
    print("=" * 60)
    print("✂️  正在根据官方 Split 文件拆分数据集 (Train/Val/Test)...")
    print("=" * 60)
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 错误: 找不到文件 {INPUT_CSV}")
        return

    # 1. 加载数据
    print(f"1. 加载你的数据集: {INPUT_CSV}")
    df_my = pd.read_csv(INPUT_CSV)
    
    print(f"2. 加载官方 Split: {OFFICIAL_SPLIT}")
    df_official = pd.read_csv(OFFICIAL_SPLIT)
    
    # 2. 获取官方定义的 study_id 集合
    train_studies = set(df_official[df_official['split'] == 'train']['study_id'])
    val_studies   = set(df_official[df_official['split'] == 'validate']['study_id'])
    test_studies  = set(df_official[df_official['split'] == 'test']['study_id'])
    
    print(f"   -> 官方定义: Train={len(train_studies)}, Val={len(val_studies)}, Test={len(test_studies)}")
    
    # 3. 执行拆分
    df_train = df_my[df_my['study_id'].isin(train_studies)]
    df_val   = df_my[df_my['study_id'].isin(val_studies)]
    df_test  = df_my[df_my['study_id'].isin(test_studies)]
    
    # 4. 保存
    print("-" * 60)
    print(f"📦 拆分结果:")
    
    df_train.to_csv(OUTPUT_TRAIN, index=False)
    print(f"   ✅ [TRAIN] 保存至: {OUTPUT_TRAIN} ({len(df_train)} 条)")
    
    df_val.to_csv(OUTPUT_VAL, index=False)
    print(f"   ✅ [VAL]   保存至: {OUTPUT_VAL}   ({len(df_val)} 条)")
    
    df_test.to_csv(OUTPUT_TEST, index=False)
    print(f"   ✅ [TEST]  保存至: {OUTPUT_TEST}  ({len(df_test)} 条)")
    
    print("-" * 60)
    print("🎉 完成！现在可以在训练代码中直接加载对应的CSV了。")

if __name__ == "__main__":
    split_by_official_list_v2()