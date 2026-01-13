import pandas as pd
import os
from collections import defaultdict

# ==============================================================================
# --- 
# 配置区域：请在此处修改您的路径 
#从 admissions.csv (MIMIC-IV 病人住院信息) 和 mimic-cxr-2.0.0-metadata.csv (MIMIC-CXR 影像元数据) 两个文件中提取共有的 subject_id
# 然后按照规则分组并保存到不同的文件里。
# ---

# ==============================================================================

# 1. 您电脑上 admissions.csv 文件的完整路径
#    请确保路径分隔符是正确的 (在Windows上使用 / 或者 \\)
PATH_TO_ADMISSIONS_CSV = "D:/PaperProject/KG_Contrast/data/mimic-iv/admissions.csv"

# 2. 【新增】您电脑上 mimic-cxr-2.0.0-metadata.csv 文件的完整路径
PATH_TO_METADATA_CSV = "D:/PaperProject/KG_Contrast/data/mimic-cxr/mimic-cxr-2.0.0-metadata.csv"

# 3. 您希望将分类好的【共有ID】文件存放在哪个文件夹
#    为了避免覆盖原先的结果，建议使用一个新的文件夹名
OUTPUT_DIRECTORY = "D:/PaperProject/KG_Contrast/data/common_subject_id_groups"

# ==============================================================================
# --- 主程序代码：通常无需修改以下内容 ---
# ==============================================================================

def main():
    """
    主执行函数
    """
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"输出目录已确认: '{OUTPUT_DIRECTORY}'")
    
    # --- 代码修改部分：读取两个文件并找到交集 ---
    
    # 1. 读取 admissions.csv 并获取唯一的 subject_id
    try:
        print(f"正在从 '{PATH_TO_ADMISSIONS_CSV}' 读取数据...")
        # 为了提高效率，只读取 'subject_id' 这一列
        df_admissions = pd.read_csv(PATH_TO_ADMISSIONS_CSV, usecols=['subject_id'], dtype={'subject_id': str})
        # 使用集合(set)来存储ID，方便快速计算交集
        admissions_ids = set(df_admissions['subject_id'].unique())
        print(f"✅ 读取完成。从 admissions.csv 中找到 {len(admissions_ids)} 个唯一的 subject_id。")
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{PATH_TO_ADMISSIONS_CSV}'。请检查路径是否正确。")
        return

    # 2. 读取 mimic-cxr-2.0.0-metadata.csv 并获取唯一的 subject_id
    try:
        print(f"正在从 '{PATH_TO_METADATA_CSV}' 读取数据...")
        df_metadata = pd.read_csv(PATH_TO_METADATA_CSV, usecols=['subject_id'], dtype={'subject_id': str})
        metadata_ids = set(df_metadata['subject_id'].unique())
        print(f"✅ 读取完成。从 metadata.csv 中找到 {len(metadata_ids)} 个唯一的 subject_id。")
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{PATH_TO_METADATA_CSV}'。请检查路径是否正确。")
        return

    # 3. 计算两个集合的交集，得到共有的 subject_id
    print("\n正在计算两个文件共有的 subject_id...")
    # set.intersection() 可以高效地找出两个集合中共同的元素
    common_subject_ids = sorted(list(admissions_ids.intersection(metadata_ids)))
    
    if not common_subject_ids:
        print("❌ 两个文件中没有找到任何共有的 subject_id。程序即将退出。")
        return
        
    print(f"✅ 共找到 {len(common_subject_ids)} 个共有的 subject_id。")
    print("-" * 50)

    # --- 后续逻辑不变，处理的对象从 unique_subject_ids 变为 common_subject_ids ---

    # 按前缀 (p10, p11, ...) 对【共有ID】进行分组
    grouped_ids = defaultdict(list)

    print("正在按 'p' + 前两位数字的前缀对【共有ID】进行分组...")
    # 【修改】现在遍历的是共有的ID列表
    for sid in common_subject_ids:
        if len(sid) >= 2:
            # 例如: subject_id '10000032' 的前缀是 'p10'
            prefix = f"p{sid[:2]}"
            grouped_ids[prefix].append(sid)

    print("✅ 分组完成。")
    print("-" * 50)

    # 将每个分组写入到单独的文件中
    print("正在将每个分组写入到不同的文件中...")
    for prefix, ids in sorted(grouped_ids.items()): # 按p10, p11...排序
        # 为每个文件内的ID进行排序，这是一个好习惯
        ids.sort()
        
        # 构建输出文件名，例如: 'subject_ids_p10.txt'
        output_filename = os.path.join(OUTPUT_DIRECTORY, f"subject_ids_{prefix}.txt")
        
        with open(output_filename, 'w') as f:
            for sid in ids:
                f.write(sid + '\n')
                
        print(f"  -> 已保存 {len(ids):>6} 个ID到文件: '{output_filename}'")

    print("-" * 50)
    print("🎉 全部完成！")
    print(f"所有分类好的【共有 subject_id】文件都已保存在目录中: '{OUTPUT_DIRECTORY}'")

if __name__ == "__main__":
    main()
