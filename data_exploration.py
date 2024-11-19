import pandas as pd
import numpy as np
import json
import argparse
import os
#Import any other necessary modules


def load_data(path):
    df = pd.read_csv(path,low_memory=False)
    return df


def get_highest_correlated_feature(df):
    data = pd.get_dummies(df, columns=['cat_feature_1', 'cat_feature_2'], drop_first=True)
    correlation_matrix = data.corr()
    target_correlations = correlation_matrix['target'].abs().drop('target')
    return target_correlations.idxmax()



def category_with_highest_mean_cat_feature_2(df):
    target_means = df.groupby('cat_feature_2')['target'].mean()
    return target_means.max()



def abs_std_dev_diff_btwn_groups_cat_feature_1(df):
    std_devs = df.groupby('cat_feature_1')['target'].std()
    return abs(std_devs.diff().iloc[-1])



def min_feature_8_for_cat_feature_2(df):
    return df[df['cat_feature_2'] == 'Category_C']['feature_8'].min()



def get_variance_feature_12_for_group(df):
    return df[(df['cat_feature_1'] == 'High') & (df['cat_feature_2'] == 'Category_A')]['feature_12'].var()




def main():
    cur_dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',type=str,default=cur_dir)
    parser.add_argument('--data_dir',type=str,default='')
    parser.add_argument('--dataset',type=str,default='case_study_data.csv')
    parser.add_argument('--output_dir',type=str,default='output')
    parser.add_argument('--output_name',type=str,default="calculated_metrics.json")
    args = parser.parse_args()

    data_path = os.path.join(cur_dir,args.data_dir,args.dataset)
    df = load_data(data_path)
    results = {}
    results['get_highest_correlated_feature'] = get_highest_correlated_feature(df)
    results['category_with_highest_mean_cat_feature_2'] = category_with_highest_mean_cat_feature_2(df)
    results['abs_std_dev_diff_btwn_groups_cat_feature_1'] = round(abs_std_dev_diff_btwn_groups_cat_feature_1(df),2)
    results['min_feature_8_for_cat_feature_2'] = round(min_feature_8_for_cat_feature_2(df),2)
    results['get_variance_feature_12_for_group'] = round(get_variance_feature_12_for_group(df),2)

    output_path = os.path.join(args.base_dir, args.output_dir, args.output_name)
    with open(output_path, "w") as outfile: 
        json.dump(results, outfile)


if __name__ == '__main__':
    main()