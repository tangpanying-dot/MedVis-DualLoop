import os
import shutil

# 配置路径
keep_ids_file = 'data/common_subject_id_groups/subject_ids_p10.txt'
images_dir = 'data/mimic-cxr/files/p10'

# 读取要保留的ID列表
print("正在读取ID列表...")
with open(keep_ids_file, 'r') as f:
    keep_ids = set(line.strip() for line in f if line.strip())

print(f"要保留的ID数量: {len(keep_ids)}")
print(f"示例ID: {list(keep_ids)[:5]}\n")

# 获取所有文件夹
if not os.path.exists(images_dir):
    print(f"错误: 目录 {images_dir} 不存在")
    exit(1)

folders = [f for f in os.listdir(images_dir) 
           if os.path.isdir(os.path.join(images_dir, f))]

print(f"找到 {len(folders)} 个文件夹\n")

# 分析哪些会被删除
to_delete = []
to_keep = []

for folder in folders:
    if folder.startswith('p'):
        folder_id = folder[1:]  # 去掉 'p' 前缀
        
        if folder_id in keep_ids:
            to_keep.append(folder)
        else:
            to_delete.append((folder, folder_id))
    else:
        print(f"⚠️  警告: 文件夹 '{folder}' 不符合预期格式（应以'p'开头）")

# 显示统计
print("=" * 70)
print("📊 删除前统计:")
print("=" * 70)
print(f"📁 总文件夹数量:    {len(folders)}")
print(f"✅ 将保留的文件夹:  {len(to_keep)}")
print(f"❌ 将删除的文件夹:  {len(to_delete)}")
print(f"📊 删除比例:        {len(to_delete)/len(folders)*100:.1f}%")
print("=" * 70)
print()

if len(to_delete) == 0:
    print("✅ 没有需要删除的文件夹！")
    exit(0)

# 显示将要删除的文件夹
print(f"⚠️  以下 {len(to_delete)} 个文件夹将被删除:")
print("-" * 70)
for idx, (folder, folder_id) in enumerate(to_delete[:20], 1):  # 只显示前20个
    print(f"  {idx:4d}. 文件夹: {folder:20s} | ID: {folder_id}")
if len(to_delete) > 20:
    print(f"  ... 还有 {len(to_delete) - 20} 个文件夹未显示")
print("-" * 70)
print()

# 确认操作
print("⚠️  ⚠️  ⚠️  警告: 此操作将永久删除 {} 个文件夹！⚠️  ⚠️  ⚠️".format(len(to_delete)))
confirm = input("\n确认要继续吗？输入 'YES' 继续，其他任何内容取消: ")

if confirm != 'YES':
    print("❌ 操作已取消")
    exit(0)

# 执行删除
deleted_ids = []
failed_ids = []

print("\n开始执行删除...")
print("-" * 70)

for folder, folder_id in to_delete:
    folder_path = os.path.join(images_dir, folder)
    
    try:
        shutil.rmtree(folder_path)
        deleted_ids.append(folder_id)
        print(f"✓ 已删除: {folder:20s} (ID: {folder_id})")
    except Exception as e:
        failed_ids.append((folder_id, str(e)))
        print(f"✗ 删除失败: {folder:20s} (ID: {folder_id}) - 错误: {e}")

# 显示最终统计
print()
print("=" * 70)
print("📊 删除完成统计:")
print("=" * 70)
print(f"✅ 成功删除:   {len(deleted_ids)} 个文件夹")
print(f"✗ 删除失败:   {len(failed_ids)} 个文件夹")
print(f"📁 剩余文件夹: {len(to_keep)} 个")
print("=" * 70)

if failed_ids:
    print(f"\n删除失败的ID列表（共 {len(failed_ids)} 个）:")
    print("-" * 70)
    for idx, (failed_id, error) in enumerate(failed_ids, 1):
        print(f"{idx:4d}. ID: {failed_id} - 原因: {error}")

if deleted_ids:
    print(f"\n✅ 已成功删除的ID列表（共 {len(deleted_ids)} 个）:")
    print("-" * 70)
    # 分批显示，每行5个
    for i in range(0, len(deleted_ids), 5):
        batch = deleted_ids[i:i+5]
        print("  " + ", ".join(batch))
    
print("\n" + "=" * 70)
print("✅ 操作全部完成！")
print("=" * 70)