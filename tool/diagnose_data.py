# diagnose_data.py - 快速诊断数据问题
import json
import sys

def diagnose_jsonl(file_path: str):
    """诊断JSONL文件的数据质量"""
    print("="*60)
    print(f"🔍 诊断文件: {file_path}")
    print("="*60)
    
    stats = {
        'total_lines': 0,
        'valid_lines': 0,
        'empty_real': 0,
        'empty_gen': 0,
        'very_short_real': 0,
        'very_short_gen': 0,
        'real_lengths': [],
        'gen_lengths': []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            stats['total_lines'] += 1
            
            try:
                data = json.loads(line)
                real_report = data.get('real_report', '')
                gen_report = data.get('generated_report', '')
                
                # 处理real_report可能是JSON字符串的情况
                if isinstance(real_report, str):
                    try:
                        real_dict = json.loads(real_report)
                        if isinstance(real_dict, dict):
                            real_report = real_dict.get('findings', real_report)
                    except:
                        pass
                
                # 统计长度
                real_len = len(str(real_report).strip())
                gen_len = len(str(gen_report).strip())
                
                stats['real_lengths'].append(real_len)
                stats['gen_lengths'].append(gen_len)
                
                if real_len == 0:
                    stats['empty_real'] += 1
                elif real_len < 20:
                    stats['very_short_real'] += 1
                
                if gen_len == 0:
                    stats['empty_gen'] += 1
                elif gen_len < 20:
                    stats['very_short_gen'] += 1
                
                if real_len > 0 and gen_len > 0:
                    stats['valid_lines'] += 1
                
                # 打印前3个样本
                if line_num <= 3:
                    print(f"\n📝 样本 {line_num}:")
                    print(f"  study_id: {data.get('study_id')}")
                    print(f"  real_report长度: {real_len}")
                    print(f"  generated_report长度: {gen_len}")
                    print(f"  real_report前100字符: {str(real_report)[:100]}")
                    print(f"  generated_report前100字符: {str(gen_report)[:100]}")
                    
            except Exception as e:
                print(f"\n❌ 第{line_num}行解析失败: {e}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("📊 统计结果")
    print("="*60)
    print(f"总行数: {stats['total_lines']}")
    print(f"有效样本: {stats['valid_lines']}")
    print(f"\n❌ 问题统计:")
    print(f"  real_report为空: {stats['empty_real']}")
    print(f"  generated_report为空: {stats['empty_gen']}")
    print(f"  real_report过短(<20字符): {stats['very_short_real']}")
    print(f"  generated_report过短(<20字符): {stats['very_short_gen']}")
    
    if stats['real_lengths']:
        import numpy as np
        print(f"\n📏 长度分布 (real_report):")
        print(f"  平均: {np.mean(stats['real_lengths']):.1f}")
        print(f"  中位数: {np.median(stats['real_lengths']):.1f}")
        print(f"  最小: {np.min(stats['real_lengths'])}")
        print(f"  最大: {np.max(stats['real_lengths'])}")
        
        print(f"\n📏 长度分布 (generated_report):")
        print(f"  平均: {np.mean(stats['gen_lengths']):.1f}")
        print(f"  中位数: {np.median(stats['gen_lengths']):.1f}")
        print(f"  最小: {np.min(stats['gen_lengths'])}")
        print(f"  最大: {np.max(stats['gen_lengths'])}")
    
    # 给出建议
    print("\n" + "="*60)
    print("💡 诊断建议")
    print("="*60)
    
    if stats['valid_lines'] == 0:
        print("❌ 严重问题: 没有有效样本!")
        print("   请检查数据格式是否正确")
    elif stats['valid_lines'] < stats['total_lines'] * 0.5:
        print("⚠️  超过50%的样本无效")
        print(f"   空的real_report: {stats['empty_real']}")
        print(f"   空的generated_report: {stats['empty_gen']}")
    elif stats['very_short_gen'] > stats['valid_lines'] * 0.3:
        print("⚠️  超过30%的generated_report过短")
        print("   这可能导致CE评估效果不好")
    else:
        print("✅ 数据质量良好!")
        print(f"   有效样本率: {stats['valid_lines']/stats['total_lines']*100:.1f}%")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python diagnose_data.py <jsonl文件路径>")
        print("示例: python diagnose_data.py report/preds_sample-all.jsonl")
        sys.exit(1)
    
    diagnose_jsonl(sys.argv[1])