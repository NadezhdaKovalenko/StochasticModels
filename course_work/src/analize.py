import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import seaborn as sns

PATH_TO_DATA_FILE = '../data/data_new.xlsx'
PLOTS_DIR = 'results'
DEFAULT_IMAGE_EXT = 'png'

def get_samples_dfs_from_file(listName, filename=PATH_TO_DATA_FILE):
    xl_file = pd.ExcelFile(filename)

    samples_dfs = {sheet_name: xl_file.parse(sheet_name)
                   for sheet_name in xl_file.sheet_names}
    return samples_dfs[listName]

def drop_extra_columns(df):
    return df.drop(columns=['№ п.п', 'проба', 'метан'])

def fill_missing_values(df, fill_strategy='median'):
    tmp_df = df.apply(pd.to_numeric, errors='coerce')
    return tmp_df.fillna(getattr(tmp_df, fill_strategy)())

def scale_data(df, Scaler=StandardScaler):
    return pd.DataFrame(Scaler().fit_transform(df))

def matrix_covv(data):
    cov_matrix = np.cov(data.T)
    print("cov_matrix shape:",cov_matrix.shape)
    print("Covariance_matrix",cov_matrix)
    return cov_matrix

def drop_outliers_with_IQR(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_filtered

def draw_samples_on_scatter_plot(PCs_df, sample_1_size, sample_2_size, description):
    plt.scatter(x=PCs_df.iloc[:sample_1_size, 0],
                    y=PCs_df.iloc[:sample_1_size, 1],
                    c='b',
                    label='sample 1',
                    alpha=0.4)
    plt.scatter(x=PCs_df.iloc[sample_1_size:sample_1_size + sample_2_size, 0],
                    y=PCs_df.iloc[sample_1_size:sample_1_size + sample_2_size, 1],
                    c='r',
                    label='sample 2',
                    alpha=0.4)
    plt.scatter(x=PCs_df.iloc[sample_1_size + sample_2_size:, 0],
                    y=PCs_df.iloc[sample_1_size + sample_2_size:, 1],
                    c='k',
                    label='sample 3',
                    alpha=0.4)

    plt.legend(loc='upper left')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Result of two given samples using PC1 and PC2 \n ({description})')
    fname = '.'.join(['scatter ' + description, DEFAULT_IMAGE_EXT])
    plt.savefig(os.path.join(PLOTS_DIR, fname))
    plt.show()

def draw_PCA_scree_plot(pca, description):
    PC_num = range(pca.n_components_)
    plt.bar(PC_num, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCs')
    plt.ylabel('explained variance ratio')
    plt.xticks(PC_num)
    plt.title(f'PCs explained variance distribution \n ({description})')
    fname = '.'.join(['scree ' + description, DEFAULT_IMAGE_EXT])
    plt.savefig(os.path.join(PLOTS_DIR, fname))
    plt.show()

def plot_heatmap(data, format, description, xLabel, yLabel):
    if len(xLabel) > 0:
        sns.heatmap(data, annot = True, fmt=format, xticklabels = xLabel, yticklabels = yLabel)
    else:
        sns.heatmap(data, annot = True, fmt=format)
    fname = '.'.join(['heatmap ' + description, DEFAULT_IMAGE_EXT])
    plt.savefig(os.path.join(PLOTS_DIR, fname))
    plt.show()

def analyze_component(pca, features):
    maxId = 0
    maxValue = 0

    for i in range(len(pca.components_)):
        for j in range(len(pca.components_[0])):
            if pca.components_[i][j] > maxValue:
                maxValue = pca.components_[i][j]
                maxId = i

    component_idx = maxId
    PC = np.array(pca.components_[component_idx])
    top_features_idxs = np.argsort(np.abs(PC))[::-1]
    top_features = np.array(features)[top_features_idxs]
    return top_features, PC[top_features_idxs], maxId

def main():
    # считывание данных
    sample_1_df = get_samples_dfs_from_file('1')
    sample_2_df = get_samples_dfs_from_file('2')
    sample_3_df = get_samples_dfs_from_file('3')
    sample_4_df = get_samples_dfs_from_file('4')

    # удаление лишний столбцов
    sample_1_df = drop_extra_columns(sample_1_df)
    sample_2_df = drop_extra_columns(sample_2_df)
    sample_3_df = drop_extra_columns(sample_3_df)
    sample_4_df = drop_extra_columns(sample_4_df)

    # получение наименований столбцов
    features = sample_1_df.columns.values

    # заполнение медианой недостоющих значений
    sample_1_df = fill_missing_values(sample_1_df)
    sample_2_df = fill_missing_values(sample_2_df)
    sample_3_df = fill_missing_values(sample_3_df)
    sample_4_df = fill_missing_values(sample_4_df)

    # удаление выбросов
    sample_1_df = drop_outliers_with_IQR(sample_1_df)
    sample_2_df = drop_outliers_with_IQR(sample_2_df)
    sample_3_df = drop_outliers_with_IQR(sample_3_df)
    sample_4_df = drop_outliers_with_IQR(sample_4_df)

    # слияние данных
    all_samples_df = sample_1_df.append(sample_2_df)
    all_samples_df = all_samples_df.append(sample_3_df)
    #all_samples_df = sample_1_df.append(sample_4_df)

    # масштабирование данных
    all_samples_df = scale_data(all_samples_df)
    n_features = len(all_samples_df.columns)  
    sample_1_size = len(sample_1_df)
    sample_2_size = len(sample_2_df)
    sample_3_size = len(sample_3_df)
    sample_4_size = len(sample_4_df)

    # применение метода главных компонент
    pca = PCA(n_components=n_features)
    pcs_df = pd.DataFrame(pca.fit_transform(all_samples_df))

    #
    description = 'Data is scaled and outliers are dropped'
    draw_PCA_scree_plot(pca, description=description)
    draw_samples_on_scatter_plot(pcs_df, sample_1_size, sample_2_size, description=description)
    
    # 
    pca_1 = PCA(n_components=n_features)
    pca_1.fit(sample_1_df)
    plot_heatmap(pca_1.components_, '.2f', 'allComponents 1 sample', features, features)

    print("PC1_data1")
    top_features_pc1, top_coeffs_pc1, maxId_pc1 = analyze_component(pca_1, features)
    dataMap = np.reshape(top_coeffs_pc1, (len(top_coeffs_pc1), 1))
    plot_heatmap(dataMap, '.8f', 'Features 1 sample', '', '')

    d1 = pd.DataFrame.from_dict({'Feature name': top_features_pc1,
                                'Features 1 sample': top_coeffs_pc1})
    print(d1)

    # 
    pca_2 = PCA(n_components=n_features)
    pca_2.fit(sample_2_df)
    plot_heatmap(pca_2.components_, '.2f', 'allComponents 2 sample', features, features)

    print("PC2")
    top_features_pc2, top_coeffs_pc2, maxId_pc2 = analyze_component(pca_2, features)
    dataMap = np.reshape(top_coeffs_pc2, (len(top_coeffs_pc2), 1))
    plot_heatmap(dataMap, '.8f', 'Features 2 sample', '', '')
    d2 = pd.DataFrame.from_dict({'Feature name': top_features_pc2,
                                'Features 2 sample': top_coeffs_pc2})
    print(d2)

    # 
    pca_3 = PCA(n_components=n_features)
    pca_3.fit(sample_3_df)
    plot_heatmap(pca_3.components_, '.2f', 'allComponents 3 sample', features, features)

    print("data3")
    top_features_pc3, top_coeffs_pc3, maxId_pc3 = analyze_component(pca_3, features)
    dataMap = np.reshape(top_coeffs_pc3, (len(top_coeffs_pc3), 1))
    plot_heatmap(dataMap, '.8f', 'Features 3 sample', '', '')

    d3 = pd.DataFrame.from_dict({'Feature name': top_features_pc3,
                                'Features 3 sample': top_coeffs_pc3})
    print(d3)


if __name__ == '__main__':
    main()