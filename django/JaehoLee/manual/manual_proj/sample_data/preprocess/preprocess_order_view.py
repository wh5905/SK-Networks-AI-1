import os

import pandas as pd

def load_data(orders_file, view_counts_file):
    orders_df = pd.read_excel(orders_file)
    view_count_df = pd.read_excel(view_counts_file)
    return orders_df, view_count_df

def merge_data(orders_df, view_counts_df):
    merged_df = pd.merge(orders_df, view_counts_df, on=['accountId', 'productId'], how='left')
    return merged_df

def preprocess_data(merged_df):
    merged_df['viewCount'] = merged_df['viewCount'].fillna(0)
    return merged_df

def export_data_to_excel(df, filePath):
    df.to_excel(filePath, index=False, engine='openpyxl')
    print(f"전처리 완료: {filePath}")

if __name__ == "__main__":
    currentDirectory = os.getcwd()
    print(f"currentDirectory: {currentDirectory}")

    orders_info_file = "../../../../../../fastapiAI/JaehoLee/fastapi_demo/assets/orders_data.xlsx"
    view_count_info_file = "../../../../../../fastapiAI/JaehoLee/fastapi_demo/assets/view_counts_data.xlsx"
    output_file = "../../preprocessed_orders_data.xlsx"

    orders_df, view_counts_df = load_data(orders_info_file, view_count_info_file)
    merged_df = merge_data(orders_df, view_counts_df)
    preprocessed_df = preprocess_data(merged_df)
    export_data_to_excel(preprocessed_df, output_file)
