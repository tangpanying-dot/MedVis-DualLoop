#!/usr/bin/env python3
"""
增量清洗脚本 - 去除剩余3.6%的噪音
基于当前的processed_dataset.csv进行额外清洗
"""
import pandas as pd
import re
from tqdm import tqdm

def clean_remaining_noise(
    input_csv='data/processed_dataset.csv',
    output_csv='data/processed_dataset_v17.csv'
):
    """
    在现有清洗基础上，去除剩余的噪音
    """
    
    print("=" * 80)
    print("🧹 增量清洗脚本 - 去除剩余3.6%噪音")
    print("=" * 80)
    
    # 加载数据
    print(f"\n[1/5] 加载数据: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"✅ 加载 {len(df)} 条记录")
    
    # 定义清洗规则
    print("\n[2/5] 定义清洗规则...")
    
    # 定义需要清洗的噪音模式
    noise_patterns = {
        'Dr./医生相关': [
            # 完整移除包含这些的句子
            r'[^.]*?(?:conveyed|relayed|discussed|communicated) by Dr\.[^.]*?\.',
            r'[^.]*?findings? (?:were|was) (?:conveyed|relayed|discussed)[^.]*?\.',
            r'[^.]*?by Dr\. \w+ to Dr\. \w+[^.]*?\.',
            r'[^.]*?Dr\. \w+[^.]*?(?:telephone|phone|pager)[^.]*?\.',
        ],
        
        '时间戳相关': [
            r'[^.]*?(?:done|obtained|performed) at \d{2}:\d{2}[^.]*?\.',
            r'[^.]*?(?:examination|study) done at[^.]*?\.',
            r'[^.]*?at \d{2}:\d{2}[^.]*?(?:hours|on)[^.]*?\.',
            r'\d{2}:\d{2}(?:\s+hours)?',  # 单独的时间戳
        ],
        
        '通讯信息': [
            r'[^.]*?(?:telephone|phone|pager)[^.]*?at \d{2}:\d{2}[^.]*?\.',
            r'[^.]*?pager was placed[^.]*?\.',
            r'[^.]*?min(?:utes)? after[^.]*?\.',
            r'[^.]*?results were conveyed[^.]*?\.',
        ],
        
        '技术/行政信息': [
            r'Analysis is performed in direct\s*',
            r'\d{2}:\d{2}\s+is submitted\.?',
            r'[^.]*?is submitted[^.]*?\.',
            r',\s*MD\s*=\s*CC:\s*DR\..*',
            r'Dictated by[^.]*?\.',
            r'Attending:[^.]*?\.',
            r'Resident:[^.]*?\.',
        ],
        
        '其他常见噪音': [
            r'\s+text\s+on\s+at\s*',  # "text on at"
            r'Findings:\s*$',  # 空的Findings标签
            r'Impression:\s*$',  # 空的Impression标签
            r'\s{2,}',  # 多余空格
        ]
    }
    
    total_patterns = sum(len(patterns) for patterns in noise_patterns.values())
    print(f"   定义了 {len(noise_patterns)} 类共 {total_patterns} 个清洗规则")
    
    # 清洗函数
    def clean_text(text):
        """清洗单个文本"""
        if pd.isna(text) or text == '':
            return text
        
        text = str(text)
        original_text = text
        
        # 应用所有清洗规则
        for category, patterns in noise_patterns.items():
            for pattern in patterns:
                text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 清理多余的句号
        text = re.sub(r'\.{2,}', '.', text)
        
        # 清理句子开头的连接词
        text = re.sub(r'^\s*(?:and|but|or|however|therefore)\s+', '', text, flags=re.IGNORECASE)
        
        return text
    
    # 清洗report列
    print("\n[3/5] 清洗'report'列...")
    print("   这可能需要几分钟...")
    
    cleaned_reports = []
    noise_count = 0
    
    for text in tqdm(df['report'], desc="   处理进度"):
        cleaned = clean_text(text)
        cleaned_reports.append(cleaned)
        
        # 统计是否有变化
        if cleaned != text:
            noise_count += 1
    
    df['report'] = cleaned_reports
    
    print(f"   ✅ 完成！发现并清洗了 {noise_count} 条记录 ({noise_count/len(df)*100:.1f}%)")
    
    # 如果有findings列，也清洗
    if 'findings' in df.columns:
        print("\n[4/5] 清洗'findings'列...")
        cleaned_findings = []
        findings_noise_count = 0
        
        for text in tqdm(df['findings'], desc="   处理进度"):
            cleaned = clean_text(text)
            cleaned_findings.append(cleaned)
            if cleaned != text:
                findings_noise_count += 1
        
        df['findings'] = cleaned_findings
        print(f"   ✅ 完成！发现并清洗了 {findings_noise_count} 条记录 ({findings_noise_count/len(df)*100:.1f}%)")
    else:
        print("\n[4/5] 跳过findings列（不存在）")
    
    # 保存
    print(f"\n[5/5] 保存清洗后的数据: {output_csv}")
    df.to_csv(output_csv, index=False)
    print(f"   ✅ 成功保存 {len(df)} 条记录")
    
    # 统计
    print("\n" + "=" * 80)
    print("📊 清洗统计")
    print("=" * 80)
    print(f"输入文件: {input_csv}")
    print(f"输出文件: {output_csv}")
    print(f"总记录数: {len(df)}")
    print(f"清洗记录数: {noise_count} ({noise_count/len(df)*100:.1f}%)")
    if 'findings' in df.columns:
        print(f"findings清洗: {findings_noise_count} ({findings_noise_count/len(df)*100:.1f}%)")
    
    print("\n✅ 清洗完成！")
    print("\n下一步:")
    print("1. 检查输出文件: data/processed_dataset_v17.csv")
    print("2. 运行质量检查: python check_training_data.py")
    print("   (记得修改脚本中的csv路径)")
    print("3. 如果满意，备份原文件并替换:")
    print("   mv data/processed_dataset.csv data/processed_dataset_backup.csv")
    print("   mv data/processed_dataset_v17.csv data/processed_dataset.csv")
    print("4. 重新训练Stage2: python train_stage2_optimized.py")
    print("=" * 80)

if __name__ == "__main__":
    clean_remaining_noise()