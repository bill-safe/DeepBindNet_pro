import pandas as pd

def remove_duplicate_rows(input_file, output_file=None):
    """
    删除 CSV 文件中重复的行。

    参数:
        input_file (str): 输入的 CSV 文件路径。
        output_file (str): 输出的 CSV 文件路径。如果为 None，则覆盖原文件。
    """
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 删除重复行
    df_cleaned = df.drop_duplicates()

    # 保存处理后的文件
    if output_file is None:
        output_file = input_file  # 覆盖原文件

    df_cleaned.to_csv(output_file, index=False)
    print(f"重复行已删除，保存至: {output_file}")

remove_duplicate_rows("sample_input.csv")