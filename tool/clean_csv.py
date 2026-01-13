#!/usr/bin/env python3
"""
delete_empty_reports_v2.py
作用: 删除空报告记录及其对应的影像文件夹和报告文件（自适应版本）

改进:
1. 根据subject_id和study_id自动构建文件路径
2. 自动检测CSV列名
3. 更强的容错能力
"""

import os
import json
import pandas as pd
import argparse
import shutil
from datetime import datetime
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='删除空报告及其关联的影像文件夹和报告文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 预览模式(不实际删除) - 推荐先运行
  python delete_empty_reports_v2.py --dry-run
  
  # 实际删除
  python delete_empty_reports_v2.py
  
  # 指定数据目录
  python delete_empty_reports_v2.py --data-dir /path/to/data
        """
    )
    parser.add_argument('--csv', type=str, default='data/processed_dataset_w.csv',
                        help='CSV文件路径')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='数据根目录')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览模式，不实际删除文件')
    parser.add_argument('--skip-backup', action='store_true',
                        help='跳过CSV备份（不推荐）')
    return parser.parse_args()

def is_report_empty(report_json_str):
    """检查报告是否为空"""
    try:
        report_dict = json.loads(report_json_str)
        if not report_dict:
            return True
        all_empty = all(not v.strip() for v in report_dict.values() 
                       if isinstance(v, str))
        return all_empty
    except:
        return True

def get_dir_size(directory):
    """计算文件夹大小（字节）"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except:
        pass
    return total_size

def get_file_size(filepath):
    """获取文件大小（字节）"""
    try:
        return os.path.getsize(filepath)
    except:
        return 0

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

class DeletionManager:
    """管理文件删除操作"""
    
    def __init__(self, data_dir, dry_run=False):
        self.data_dir = data_dir
        self.dry_run = dry_run
        self.mimic_cxr_root = os.path.join(data_dir, "mimic-cxr")
        
        # 统计信息
        self.stats = {
            'records_to_delete': 0,
            'image_folders_deleted': 0,
            'image_folders_not_found': 0,
            'report_files_deleted': 0,
            'report_files_not_found': 0,
            'total_images_in_folders': 0,
            'total_size_freed': 0
        }
        
        # 删除日志
        self.deletion_log = []
        
    def build_paths_from_ids(self, subject_id, study_id):
        """
        从subject_id和study_id构建文件路径
        
        例如: subject_id=10000032, study_id=50414267
        返回:
          - 报告路径: files/p10/p10000032/s50414267.txt
          - 影像文件夹: images/p10/p10000032/s50414267/
        """
        try:
            # 转换为整数
            sid = int(subject_id)
            stid = str(study_id)
            
            # 构建目录结构
            # p10 = "p" + subject_id的前两位数字
            p_prefix = f"p{str(sid)[:2]}"
            p_folder = f"p{sid}"
            s_name = f"s{stid}"
            
            # 报告文件路径: files/p10/p10000032/s50414267.txt
            report_path = os.path.join("files", p_prefix, p_folder, f"{s_name}.txt")
            
            # 影像文件夹路径: images/p10/p10000032/s50414267/
            image_folder = os.path.join("images", p_prefix, p_folder, s_name)
            
            return report_path, image_folder
        except Exception as e:
            print(f"⚠️  构建路径失败: subject_id={subject_id}, study_id={study_id}, error={e}")
            return None, None
    
    def get_report_file_path(self, subject_id, study_id):
        """获取报告文件完整路径"""
        report_path, _ = self.build_paths_from_ids(subject_id, study_id)
        if report_path:
            return os.path.join(self.mimic_cxr_root, report_path)
        return None
    
    def get_image_folder_path(self, subject_id, study_id):
        """获取影像文件夹完整路径"""
        _, image_folder = self.build_paths_from_ids(subject_id, study_id)
        if image_folder:
            return os.path.join(self.mimic_cxr_root, image_folder)
        return None
    
    def count_files_in_folder(self, folder_path):
        """统计文件夹中的文件数量"""
        try:
            if os.path.exists(folder_path):
                return len([f for f in os.listdir(folder_path) 
                           if os.path.isfile(os.path.join(folder_path, f))])
        except:
            pass
        return 0
    
    def delete_image_folder(self, folder_path, subject_id, study_id):
        """删除影像文件夹"""
        if not folder_path:
            self.deletion_log.append({
                'type': 'image_folder',
                'subject_id': subject_id,
                'study_id': study_id,
                'path': 'N/A',
                'status': 'path_error'
            })
            return False, 0, 0
        
        if os.path.exists(folder_path):
            # 统计信息
            num_files = self.count_files_in_folder(folder_path)
            folder_size = get_dir_size(folder_path)
            
            if not self.dry_run:
                try:
                    shutil.rmtree(folder_path)
                    self.deletion_log.append({
                        'type': 'image_folder',
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'path': folder_path,
                        'num_files': num_files,
                        'size': folder_size,
                        'status': 'deleted'
                    })
                    return True, num_files, folder_size
                except Exception as e:
                    self.deletion_log.append({
                        'type': 'image_folder',
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'path': folder_path,
                        'status': f'error: {str(e)}'
                    })
                    return False, 0, 0
            else:
                # Dry run模式
                self.deletion_log.append({
                    'type': 'image_folder',
                    'subject_id': subject_id,
                    'study_id': study_id,
                    'path': folder_path,
                    'num_files': num_files,
                    'size': folder_size,
                    'status': 'would_delete'
                })
                return True, num_files, folder_size
        else:
            self.deletion_log.append({
                'type': 'image_folder',
                'subject_id': subject_id,
                'study_id': study_id,
                'path': folder_path,
                'status': 'not_found'
            })
            return False, 0, 0
    
    def delete_report_file(self, file_path, subject_id, study_id):
        """删除报告文件"""
        if not file_path:
            self.deletion_log.append({
                'type': 'report_file',
                'subject_id': subject_id,
                'study_id': study_id,
                'path': 'N/A',
                'status': 'path_error'
            })
            return False, 0
        
        if os.path.exists(file_path):
            file_size = get_file_size(file_path)
            
            if not self.dry_run:
                try:
                    os.remove(file_path)
                    self.deletion_log.append({
                        'type': 'report_file',
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'path': file_path,
                        'size': file_size,
                        'status': 'deleted'
                    })
                    return True, file_size
                except Exception as e:
                    self.deletion_log.append({
                        'type': 'report_file',
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'path': file_path,
                        'status': f'error: {str(e)}'
                    })
                    return False, 0
            else:
                # Dry run模式
                self.deletion_log.append({
                    'type': 'report_file',
                    'subject_id': subject_id,
                    'study_id': study_id,
                    'path': file_path,
                    'size': file_size,
                    'status': 'would_delete'
                })
                return True, file_size
        else:
            self.deletion_log.append({
                'type': 'report_file',
                'subject_id': subject_id,
                'study_id': study_id,
                'path': file_path,
                'status': 'not_found'
            })
            return False, 0
    
    def process_empty_reports(self, df):
        """处理空报告"""
        print("="*80)
        print("删除空报告及关联文件")
        print("="*80)
        print(f"模式: {'🔍 预览模式 (DRY RUN)' if self.dry_run else '⚠️  实际删除模式'}")
        print("="*80)
        
        # 检查必要的列
        required_cols = ['subject_id', 'study_id', 'report']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 错误: CSV文件缺少必要的列: {missing_cols}")
            print(f"   当前列: {list(df.columns)}")
            return df
        
        # 识别空报告
        print("\n[步骤 1/4] 识别空报告记录...")
        df['is_empty'] = df['report'].apply(is_report_empty)
        empty_df = df[df['is_empty']]
        self.stats['records_to_delete'] = len(empty_df)
        
        print(f"✅ 找到 {len(empty_df):,} 条空报告记录")
        
        if len(empty_df) == 0:
            print("✅ 没有空报告需要删除")
            return df
        
        # 删除影像文件夹
        print(f"\n[步骤 2/4] 删除影像文件夹...")
        for idx, row in tqdm(empty_df.iterrows(), total=len(empty_df), desc="删除影像"):
            folder_path = self.get_image_folder_path(row['subject_id'], row['study_id'])
            success, num_files, size = self.delete_image_folder(
                folder_path, row['subject_id'], row['study_id']
            )
            
            if success:
                self.stats['image_folders_deleted'] += 1
                self.stats['total_images_in_folders'] += num_files
                self.stats['total_size_freed'] += size
            else:
                self.stats['image_folders_not_found'] += 1
        
        # 删除报告文件
        print(f"\n[步骤 3/4] 删除报告文件...")
        for idx, row in tqdm(empty_df.iterrows(), total=len(empty_df), desc="删除报告"):
            file_path = self.get_report_file_path(row['subject_id'], row['study_id'])
            success, size = self.delete_report_file(
                file_path, row['subject_id'], row['study_id']
            )
            
            if success:
                self.stats['report_files_deleted'] += 1
                self.stats['total_size_freed'] += size
            else:
                self.stats['report_files_not_found'] += 1
        
        # 更新DataFrame
        print(f"\n[步骤 4/4] 更新CSV记录...")
        df_clean = df[~df['is_empty']].copy()
        df_clean = df_clean.drop(columns=['is_empty'])
        
        return df_clean
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*80)
        print("删除统计摘要")
        print("="*80)
        
        print(f"\n📊 记录统计:")
        print(f"  空报告记录数:           {self.stats['records_to_delete']:,}")
        
        print(f"\n📁 影像文件夹:")
        print(f"  成功删除:               {self.stats['image_folders_deleted']:,}")
        print(f"  未找到:                 {self.stats['image_folders_not_found']:,}")
        print(f"  包含影像文件数:         {self.stats['total_images_in_folders']:,}")
        
        print(f"\n📄 报告文件:")
        print(f"  成功删除:               {self.stats['report_files_deleted']:,}")
        print(f"  未找到:                 {self.stats['report_files_not_found']:,}")
        
        print(f"\n💾 磁盘空间:")
        print(f"  释放空间:               {format_size(self.stats['total_size_freed'])}")
        
        print("="*80)
    
    def save_deletion_log(self, log_file='deletion_log.json'):
        """保存删除日志"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'statistics': self.stats,
            'deletion_details': self.deletion_log
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 删除日志已保存: {log_file}")

def backup_csv(csv_path):
    """备份CSV文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = csv_path.replace('.csv', f'_backup_{timestamp}.csv')
    shutil.copy2(csv_path, backup_path)
    print(f"✅ CSV备份已创建: {backup_path}")
    return backup_path

def main():
    args = parse_args()
    
    print("="*80)
    print("MIMIC-CXR 空报告清理工具 v2.0")
    print("="*80)
    print(f"CSV文件: {args.csv}")
    print(f"数据目录: {args.data_dir}")
    print(f"运行模式: {'🔍 预览模式 (不会删除任何文件)' if args.dry_run else '⚠️  删除模式'}")
    print("="*80)
    
    # 检查文件存在
    if not os.path.exists(args.csv):
        print(f"❌ 错误: CSV文件不存在 '{args.csv}'")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"❌ 错误: 数据目录不存在 '{args.data_dir}'")
        return
    
    # 备份CSV
    if not args.dry_run and not args.skip_backup:
        print("\n[准备] 备份CSV文件...")
        backup_path = backup_csv(args.csv)
    
    # 读取CSV
    print("\n[准备] 读取CSV数据...")
    df = pd.read_csv(args.csv)
    print(f"✅ 加载 {len(df):,} 条记录")
    print(f"   CSV列: {list(df.columns)}")
    
    # 创建删除管理器
    manager = DeletionManager(args.data_dir, dry_run=args.dry_run)
    
    # 处理空报告
    df_clean = manager.process_empty_reports(df)
    
    # 打印摘要
    manager.print_summary()
    
    # 保存日志
    log_file = 'deletion_log_dry_run.json' if args.dry_run else 'deletion_log.json'
    manager.save_deletion_log(log_file)
    
    # 保存清理后的CSV
    if not args.dry_run:
        print(f"\n[完成] 保存清理后的CSV...")
        output_csv = args.csv.replace('.csv', '_clean.csv')
        df_clean.to_csv(output_csv, index=False)
        print(f"✅ 清理后的CSV已保存: {output_csv}")
        print(f"   原始记录: {len(df):,}")
        print(f"   清理后记录: {len(df_clean):,}")
        print(f"   删除记录: {len(df) - len(df_clean):,}")
    else:
        print(f"\n💡 这是预览模式，没有实际删除任何文件")
        print(f"💡 如需实际删除，请运行: python {os.path.basename(__file__)}")
    
    print("\n" + "="*80)
    print("🎉 处理完成!")
    print("="*80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()