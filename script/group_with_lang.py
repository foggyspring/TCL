import torch
import torch.nn.functional as F

def cosine_similarity_matrix(x):
    """
    计算一个矩阵中所有向量之间的余弦相似度。
    参数:
    - x: 一个形状为(batch_size, features)的张量
    返回:
    - 一个形状为(batch_size, batch_size)的相似度矩阵
    """
    # 归一化向量
    x_normalized = F.normalize(x, p=2, dim=1)
    # 计算相似度矩阵
    sim_matrix = torch.mm(x_normalized, x_normalized.transpose(0, 1))
    return sim_matrix

def group_samples_by_similarity(lang_features, seg_features, threshold=0.6):
    """
    根据语音特征的相似性来分组图像样本。
    参数:
    - lang_features: 语音特征张量，形状为(batch_size, lang_feature_dim)
    - seg_features: 图像样本特征张量，形状为(batch_size, seg_feature_dim)
    - threshold: 相似性阈值
    返回:
    - 分组的索引列表，每个元素是相似样本的索引列表
    """
    # 计算相似度矩阵
    sim_matrix = cosine_similarity_matrix(lang_features)
    # 筛选相似度大于阈值的样本对
    similar_pairs = (sim_matrix > threshold).nonzero(as_tuple=False)
    
    groups = []
    for i in range(similar_pairs.shape[0]):
        # 获取相似样本对的索引
        idx1, idx2 = similar_pairs[i]
        # 检查是否已经在分组中
        already_grouped = any(idx1.item() in g for g in groups) or any(idx2.item() in g for g in groups)
        if not already_grouped:
            # 将这对样本加入到分组中
            groups.append([idx1.item(), idx2.item()])
    
    return groups

# 假设lang_features和seg_features是你的特征张量
# lang_features = ...
# seg_features = ...

# 使用函数进行分组
groups = group_samples_by_similarity(lang_features, seg_features, threshold=0.6)

# 打印分组结果
print(groups)
