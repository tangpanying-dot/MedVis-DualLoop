import os
import time

def count_dirs_at_level(root_dir, target_level):
    """
    [功能 1] 计算指定层级的文件夹总数 (保留了你之前的逻辑)
    """
    if not os.path.isdir(root_dir):
        return None
    
    current_level_dirs = [root_dir]
    
    # 逐层深入
    for level in range(target_level):
        next_level_dirs = []
        for parent_dir in current_level_dirs:
            try:
                # 扫描当前目录下的条目
                with os.scandir(parent_dir) as entries:
                    for entry in entries:
                        if entry.is_dir():
                            next_level_dirs.append(entry.path)
            except OSError:
                continue
        current_level_dirs = next_level_dirs

    return len(current_level_dirs)

def count_files_recursively(root_dir):
    """
    [功能 2] 递归扫描所有子文件夹，统计 jpg 和 txt 文件数量
    """
    print(f"\n--- 正在开始全盘文件扫描 (目标: .jpg, .txt) ---")
    print(f"📂 扫描根目录: {root_dir}")
    print("⏳ 数据量较大，请耐心等待...")

    start_time = time.time()
    
    stats = {
        'jpg': 0,
        'txt': 0,
        'others': 0
    }
    
    # os.walk 会自动遍历所有深度的子目录
    for current_root, dirs, files in os.walk(root_dir):
        for file in files:
            # 获取小写后缀名以忽略大小写差异
            ext = os.path.splitext(file)[1].lower()
            
            if ext in ['.jpg', '.jpeg']:
                stats['jpg'] += 1
            elif ext == '.txt':
                stats['txt'] += 1
            else:
                stats['others'] += 1
                
        # (可选) 简单的进度条显示，每扫描 10000 个 JPG 显示一次
        total_imgs = stats['jpg']
        if total_imgs > 0 and total_imgs % 10000 == 0 and ext in ['.jpg', '.jpeg']:
             print(f"   -> 已累计发现 {total_imgs} 张影像...")

    end_time = time.time()
    duration = end_time - start_time
    
    return stats, duration

# =========================================================
# --- 主程序入口 ---
# =========================================================
if __name__ == '__main__':
    # ⚠️ 请确认这是您的数据根目录
    # 通常是 mimic-cxr/files 或者 mimic-cxr/files/p10 等
    target_directory = r'visual/visual_features/rad_dino' 
    
    if not os.path.exists(target_directory):
        print(f"❌ 错误：找不到目录 '{target_directory}'")
    else:
        print("="*50)
        print("📊 MIMIC-CXR 数据集结构与文件统计报告")
        print("="*50)

        # 1. 统计文件夹层级结构 (宏观)
        # Level 1: 前缀文件夹 (如 p10, p11...)
        l1_count = count_dirs_at_level(target_directory, 1)
        print(f"📁 [层级 1] 分组文件夹 (pXX) 数量: {l1_count}")

        # Level 2: 病人文件夹 (如 p10000032...)
        l2_count = count_dirs_at_level(target_directory, 2)
        print(f"📁 [层级 2] 病人文件夹 (subject_id) 数量: {l2_count}")

        # Level 3: 检查文件夹 (如 s50414267...)
        l3_count = count_dirs_at_level(target_directory, 3)
        print(f"📁 [层级 3] 检查文件夹 (study_id) 数量: {l3_count}")

        # 2. 统计具体文件数量 (微观)
        file_stats, cost_time = count_files_recursively(target_directory)

        print("-" * 50)
        print("✅ 统计完成！详细结果如下：")
        print(f"⏱️  文件扫描耗时: {cost_time:.2f} 秒")
        print("-" * 50)
        print(f"🖼️  影像文件 (.jpg): {file_stats['jpg']} 个")
        print(f"📄  报告文件 (.txt): {file_stats['txt']} 个")
        print(f"📦  其他文件       : {file_stats['others']} 个")
        print("=" * 50)