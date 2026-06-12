"""
01_数据清洗与预处理
=================
对应文章 Section 2.2（Data Collection and Preprocessing）。

功能:
    1. 文本清洗：去除 URL、翻译注释、话题后缀、特殊字符
    2. 时间标准化：统一中/日/韩三国不同格式的发布时间
    3. 时区转换：日韩 Twitter 数据 UTC → UTC+9 (Asia/Tokyo, Asia/Seoul)
    4. 时间特征提取：hour_of_day, day_of_week, season

输入:
    - CHI: D:\\summer_research\\投稿\\code_media\\aligned_annotations.xlsx
    - JA:  D:\\summer_research\\投稿\\code_media\\all_data\\raw_data\\JA_combined_data.xlsx
    - KO:  D:\\summer_research\\投稿\\code_media\\all_data\\raw_data\\KO_combined_data.xlsx

输出:
    - 同目录下 *_cleaned.xlsx（在原文件名基础上增加 _cleaned 后缀）
"""

import pandas as pd
import re
import os
from datetime import timedelta

# ==========================================
# 1. 核心清洗函数
# ==========================================

def clean_text_logic(text):
    """对单条帖子文本执行清洗流水线。

    处理步骤（按顺序）:
        1. 空值检查 → 返回空字符串
        2. 去除反斜杠（escape 字符残余）
        3. 去除翻译注释（如 "(翻译说明: ...)"）
        4. 去除 URL（http/https/www）
        5. 去除小红书话题后缀（"[话题]"）
        6. 去除 # 号（替换为空格）
        7. 去除 "nan" 字符串（pandas 空值转字符串残留）
        8. 合并多余空白字符

    Args:
        text: 原始文本（可能为 NaN）

    Returns:
        清洗后的纯文本字符串
    """
    if pd.isna(text):
        return ""
    text = str(text)

    # 去除反斜杠（JSON转义残留）
    text = text.replace('\\', '')

    # 去除翻译注释: (翻译说明: ...) 或 （注：...）
    text = re.sub(r'[\(（]\s*(?:翻译说明|注|翻译|译)\s*[：:].*?[\)）]', '', text, flags=re.DOTALL)

    # 去除 URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 去除小红书话题后缀
    text = text.replace('[话题]', '')
    # 去除 # 号（替换为空格，避免词粘连）
    text = text.replace('#', ' ')
    # 去除 "nan" 字符串（pandas NaN → str 残留）
    text = re.sub(r'\bnan\b', '', text, flags=re.IGNORECASE)

    # 合并连续空白字符为单个空格，去除首尾空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_season(month):
    """根据月份返回气象季节（北半球标准）。

    Args:
        month: 整数月份 (1-12)

    Returns:
        "Spring" | "Summer" | "Autumn" | "Winter"
    """
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'


# ==========================================
# 2. 单文件处理主逻辑
# ==========================================

def process_dataset(file_path, region):
    """读取一个国家的原始数据，执行清洗、时间处理、保存。

    处理流程:
        A. 列重命名: '参考内容'→'full_text', '时间'→'time', 'id'→'ID'
        B. 文本清洗: 调用 clean_text_logic
        C. 时间解析:
           - 中国: 格式 "%Y/%m/%d %H:%M:%S"，解析失败时模糊匹配兜底
           - 日韩: Twitter 格式 "%a %b %d %H:%M:%S %z %Y"，UTC→UTC+9
        D. 时间特征提取: hour_of_day, day_of_week, season
        E. 保存为 *_cleaned.xlsx

    Args:
        file_path: 原始 Excel 文件路径
        region: 'CHI' | 'JA' | 'KO'，决定时间解析策略
    """
    print(f"正在处理 [{region}] 数据: {file_path} ...")

    try:
        # --- A. 读取并重命名列 ---
        df = pd.read_excel(file_path)

        rename_map = {}
        if '参考内容' in df.columns:
            rename_map['参考内容'] = 'full_text'
        if '时间' in df.columns:
            rename_map['时间'] = 'time'
        # 统一 ID 列名为大写（部分文件可能是小写 id）
        if 'id' in df.columns:
            rename_map['id'] = 'ID'

        df.rename(columns=rename_map, inplace=True)

        # 检查关键列是否存在
        if 'full_text' not in df.columns or 'time' not in df.columns:
            print(f"错误：在文件 {file_path} 中未找到 full_text 或 time 列。当前列名: {df.columns.tolist()}")
            return

        # --- B. 文本清洗 ---
        print("  - 执行文本清洗...")
        df['full_text'] = df['full_text'].apply(clean_text_logic)

        # --- C. 时间解析与时区转换 ---
        print("  - 执行时间格式化与时区转换...")

        # 先统一转为字符串（防止 Excel 读取为 float/object 混合导致报错）
        df['time'] = df['time'].astype(str)

        if region == 'CHI':
            # 中国数据格式示例: "2023/10/7 20:06:19"
            # errors='coerce' 表示解析失败时设为 NaT（而非抛异常）
            df['dt_obj'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')

            # 补救措施：对于解析失败的行（格式不完全统一的），尝试模糊解析
            if df['dt_obj'].isna().any():
                print("    ! 注意：CHI 数据中存在非标准格式，正在尝试自动修正...")
                mask = df['dt_obj'].isna()
                # 排除原本就是空值的情况
                non_empty_mask = mask & (df['time'] != 'nan') & (df['time'] != '')
                if non_empty_mask.any():
                    df.loc[non_empty_mask, 'dt_obj'] = pd.to_datetime(
                        df.loc[non_empty_mask, 'time'], errors='coerce'
                    )

        else:
            # 日韩数据 (Twitter API 格式): "Mon Nov 20 16:14:44 +0000 2023"
            # %a: 星期缩写  %b: 月份缩写  %d: 日  %H:%M:%S: 时分秒  %z: 时区偏移  %Y: 年
            twitter_format = '%a %b %d %H:%M:%S %z %Y'

            # 指定 format 消除警告并大幅提升解析速度
            df['dt_obj'] = pd.to_datetime(df['time'], format=twitter_format, utc=True, errors='coerce')

            # 时区转换: UTC → UTC+9 (Asia/Tokyo, Asia/Seoul)
            # NaT 行会被 tz_convert 自动跳过
            df['dt_obj'] = df['dt_obj'].dt.tz_convert('Asia/Tokyo')

        # 检查解析失败数量
        failed_count = df['dt_obj'].isna().sum()
        if failed_count > 0:
            print(f"    ! 警告：有 {failed_count} 行时间解析失败，已设为 NaT (可能原数据为空或格式异常)")

        # --- D. 时间特征提取 ---
        # 小时 (0-23)
        df['hour_of_day'] = df['dt_obj'].dt.hour

        # 星期几 (Monday, Tuesday, ...)
        df['day_of_week'] = df['dt_obj'].dt.day_name()

        # 季节
        df['season'] = df['dt_obj'].dt.month.apply(get_season)

        # 将 time 列更新为清洗后的标准时间字符串（方便人工查看）
        df['time'] = df['dt_obj'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # 删除临时的 datetime 对象列
        df.drop(columns=['dt_obj'], inplace=True)

        # --- E. 保存输出 ---
        # 输出路径: 在原文件名基础上增加 _cleaned 后缀
        dir_name, file_name = os.path.split(file_path)
        base_name, ext = os.path.splitext(file_name)
        output_path = os.path.join(dir_name, f"{base_name}_cleaned{ext}")

        df.to_excel(output_path, index=False)
        print(f"  √ 处理完成！已保存至: {output_path}\n")

    except Exception as e:
        print(f"  × 处理文件 {file_path} 时出错: {e}\n")


# ==========================================
# 3. 执行入口
# ==========================================

if __name__ == "__main__":
    # 三国原始数据文件配置
    file_configs = [
        {
            "path": r"D:\summer_research\投稿\code_media\aligned_annotations.xlsx",
            "region": "CHI"
        },
        {
            "path": r"D:\summer_research\投稿\code_media\all_data\raw_data\JA_combined_data.xlsx",
            "region": "JA"
        },
        {
            "path": r"D:\summer_research\投稿\code_media\all_data\raw_data\KO_combined_data.xlsx",
            "region": "KO"
        }
    ]

    for config in file_configs:
        if os.path.exists(config["path"]):
            process_dataset(config["path"], config["region"])
        else:
            print(f"错误：找不到文件 {config['path']}，请检查路径是否正确。")
