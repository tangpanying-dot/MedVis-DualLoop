import pandas as pd
import os

# ==============================================================================
# --- 配置区域：请在这里修改您的文件路径 ---\
# ==============================================================================

# 1. 输入文件：您的原始CSV文件路径
INPUT_CSV_PATH = "D:/PaperProject/KG_Contrast/data/processed_dataset.csv"

# 2. 输出文件：存放唯一subject_id的新文件名和路径
#    建议将它和您的输入文件放在同一个目录下，方便管理
OUTPUT_TXT_PATH = "D:/PaperProject/KG_Contrast/data/unique_subject_ids.txt"

# ==============================================================================
# --- 主程序代码：通常无需修改以下内容 ---\
# ==============================================================================

def main():
    """
    主执行函数，用于读取CSV，提取、计数并保存唯一的subject_id。
    """
    # 步骤 1: 检查输入文件是否存在
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"❌ 错误：找不到输入文件 '{INPUT_CSV_PATH}'。")
        print("请检查上面的 INPUT_CSV_PATH 变量是否设置正确。")
        return

    print(f"正在从 '{os.path.basename(INPUT_CSV_PATH)}' 读取数据...")
    
    try:
        # 步骤 2: 高效读取 'subject_id' 列
        df = pd.read_csv(INPUT_CSV_PATH, usecols=['subject_id'], dtype=str)
        print("文件读取完成。")
        
        # 步骤 3: 获取唯一的 subject_id 列表
        # .unique() 返回一个包含所有唯一值的数组
        unique_ids = df['subject_id'].unique()
        
        # 这是一个好习惯：对ID进行排序，使输出文件内容更有序
        sorted_unique_ids = sorted(unique_ids)
        
        unique_id_count = len(sorted_unique_ids)
        
        print(f"已找到 {unique_id_count} 个唯一的 subject_id。")
        print("-" * 60)
        
        # 步骤 4: 将唯一的ID写入到输出文件
        print(f"正在将这些唯一的ID写入到文件: '{os.path.basename(OUTPUT_TXT_PATH)}'...")
        with open(OUTPUT_TXT_PATH, 'w') as f:
            for subject_id in sorted_unique_ids:
                f.write(subject_id + '\n')
        
        print("文件写入完成。")
        print("-" * 60)
        print("🎉 全部任务完成！")
        print(f"   -> 计数结果: {unique_id_count} 个唯一ID。")
        print(f"   -> 输出文件: 已保存在 '{OUTPUT_TXT_PATH}'")

    except ValueError:
        print(f"❌ 错误：在文件 '{INPUT_CSV_PATH}' 中未能找到名为 'subject_id' 的列。")
        print("请确认您的CSV文件包含一个列头为 'subject_id' 的列。")
    except Exception as e:
        print(f"❌ 发生了一个意料之外的错误: {e}")

if __name__ == "__main__":
    # 确保pandas已安装
    try:
        import pandas
    except ImportError:
        print("❌ 错误: 需要使用 pandas 库。")
        print("请通过命令 'pip install pandas' 来安装它。")
    else:
        main()
