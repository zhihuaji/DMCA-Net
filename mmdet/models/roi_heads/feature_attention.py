import torch.nn as nn
import torch
import random


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)

    def select_qk_tensor(self, q_tensor, predict_labels, top_k, num_imgs):
        # 每张图像中选择，非batch中选择，q_tensor; 选择的规则为: 例如predict_labels第一个为1，则在top_k找出所有值为1的位置索引，根据这些索引找到对应的q_tensor注意找不到时对应位置为全为1矩阵
        predict_num = [predict_labels[i].size(0) for i in range(num_imgs)]
        top_k_num = [top_k[i].size(0) for i in range(num_imgs)]
        if sum(top_k_num) != q_tensor.size(0):
            raise ValueError("The sum of top_k unequals the size of the qk_tensor.")

        q_tensors = []
        start_index = 0
        for i in range(num_imgs):      # 将q转换为和predict_labels大小相同的list
            q_tensors.append(q_tensor[start_index:start_index+top_k_num[i]])
            start_index = start_index+top_k_num[i]

        selected_q_temp = []
        for i in range(num_imgs):
            result = []
            for value in predict_labels[i]:
                indices = (top_k[i] == value).nonzero().view(-1)
                if indices.numel() > 0:
                    selected_index = random.choice(indices.tolist())
                    result.append(selected_index)
                else:
                    result.append(-2)

            selected_indices = torch.tensor(result).to(q_tensor[i].device)
            mask_Ture = (selected_indices != -2).clone().detach().to(q_tensor[i].device)
            selected_indices[selected_indices == -2] = 0  # 将 -2 替换为 0，以避免越界
            unmatched_tensor = torch.ones_like(q_tensor[i]).to(q_tensor[i].device)  # 未匹配的用1代替
            selected_q_tensor = torch.where(mask_Ture.unsqueeze(1).unsqueeze(1).unsqueeze(1).bool(), q_tensors[i][selected_indices.int()], unmatched_tensor)

            selected_q_temp.append(selected_q_tensor)
        selected_q_result = torch.concat(selected_q_temp, dim=0)
        if sum(predict_num) != selected_q_result.size(0):
            raise ValueError("The sum of predict_num unequals the size of the select_qk_tensor.")
        return selected_q_result


    def self_attn(self, *feature):
        # [batch_size, num_patches + 1, total_embed_dim]
        _, N, C = feature[0].shape
        abnormal_qkv = self.qkv(feature[0]).reshape(-1, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        if self.training:
            normal_qkv = self.qkv(feature[1]).reshape(-1, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            a_q, a_k, a_v = abnormal_qkv[0], abnormal_qkv[1], abnormal_qkv[2]
            n_q, n_k, n_v = normal_qkv[0], normal_qkv[1], normal_qkv[2]

            a_self_attn = (a_q @ a_k.transpose(-2, -1) * self.scale).softmax(dim=-1)
            n_self_attn = (n_q @ n_k.transpose(-2, -1) * self.scale).softmax(dim=-1)

            abnormal_self_feature = (a_self_attn @ a_v).transpose(1, 2).reshape(-1, N, C)
            normal_self_feature = (n_self_attn @ n_v).transpose(1, 2).reshape(-1, N, C)

            abnormal_self_feature = self.proj(abnormal_self_feature)
            normal_self_feature = self.proj(normal_self_feature)
            return abnormal_self_feature, normal_self_feature

        else:
            a_q, a_k, a_v = abnormal_qkv[0], abnormal_qkv[1], abnormal_qkv[2]
            a_self_attn = (a_q @ a_k.transpose(-2, -1) * self.scale).softmax(dim=-1)
            abnormal_self_feature = (a_self_attn @ a_v).transpose(1, 2).reshape(-1, N, C)
            abnormal_self_feature = self.proj(abnormal_self_feature)
            return abnormal_self_feature

    def cros_attn(self, abnormal_feature, normal_feature, num_imgs, abnormal_predict,  normal_top_k):
        _, N, C = normal_feature.shape

        abnormal_qkv = self.qkv(abnormal_feature).reshape(-1, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        normal_qkv = self.qkv(normal_feature).reshape(-1, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        a_k, a_v = abnormal_qkv[1], abnormal_qkv[2]

        select_n_q_tensor = self.select_qk_tensor(normal_qkv[0],  abnormal_predict, normal_top_k, num_imgs)
        # 加入要选normal的q和k，根据abnormal_predict对应位置的值找到normal_top_k对应的位置，然后将对应位置的q和k组成一个新的tensor
        n_a_cros_attn = (select_n_q_tensor @ a_k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        n_a_cross_feature = (n_a_cros_attn @ a_v).transpose(1, 2).reshape(-1, N, C)
        n_a_cross_feature = self.proj(n_a_cross_feature)
        return n_a_cross_feature

    def mlp(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def forward(self, *features, abnormal_predict=None, normal_top_k=None):  #

        abnormal_feature = features[0]
        abnormal_feature_norm = self.norm1(abnormal_feature)

        if self.training:
            normal_feature = features[1]
            normal_feature_norm = self.norm1(normal_feature)
            num_imgs = len(abnormal_predict)

            abnormal_feature, normal_feature = \
                self.self_attn(abnormal_feature_norm, normal_feature_norm)

            n_a_feature = self.cros_attn(abnormal_feature_norm, normal_feature_norm, num_imgs, abnormal_predict, normal_top_k)

            abnormal_feature = abnormal_feature + features[0]
            normal_feature = normal_feature + features[1]
            n_a_feature = n_a_feature + features[2]

            abnormal_feature = abnormal_feature + self.mlp(self.norm2(abnormal_feature))
            normal_feature = normal_feature + self.mlp(self.norm2(normal_feature))
            n_a_feature = n_a_feature + self.mlp(self.norm2(n_a_feature))
            return abnormal_feature, normal_feature, n_a_feature

        else:
            abnormal_feature = self.self_attn(abnormal_feature_norm)
            abnormal_feature = abnormal_feature + features[0]
            abnormal_feature = abnormal_feature + self.mlp(self.norm2(abnormal_feature))
            return abnormal_feature


class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.num_heads = 8
        self.len_channals = 256
        self.embed_dim = 49
        head_dim = self.len_channals // self.num_heads
        self.scale = head_dim ** -0.5
        self.mlp_ratio = 4.0
        self.pos_embed = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, self.embed_dim, self.len_channals)), std=0.02)
        # 初始化 self.self_cross_attn_blocks
        self.self_cross_attn_blocks = Block(dim=self.len_channals, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=True)

    def forward(self, stage0_abnormal_results, stage0_normal_results=None):

        if self.training:
            abnormal_feature = stage0_abnormal_results["roi_feature"].flatten(2).transpose(1, 2)  # [num_props, 49, 256]
            abnormal_feature = abnormal_feature + self.pos_embed  # 加入位置编码

            abnormal_predict = [torch.argmax(predict, dim=1) for predict in stage0_abnormal_results["predict_labels"]]
            normal_top_k = stage0_normal_results["top_ind"]

            normal_feature = stage0_normal_results["roi_feature"].flatten(2).transpose(1, 2)
            normal_feature = normal_feature + self.pos_embed

            n_a_feature = torch.zeros_like(abnormal_feature)

            abnormal_feature, normal_feature, n_a_feature =\
                self.self_cross_attn_blocks(abnormal_feature, normal_feature, n_a_feature,
                                            abnormal_predict=abnormal_predict,
                                            normal_top_k=normal_top_k)
            abnormal_feature = abnormal_feature.transpose(1, 2).view(-1, 256, 7, 7)
            normal_feature = normal_feature.transpose(1, 2).view(-1, 256, 7, 7)
            n_a_feature = n_a_feature.transpose(1, 2).view(-1, 256, 7, 7)
            return abnormal_feature, normal_feature, n_a_feature
        else:
            abnormal_feature = stage0_abnormal_results.flatten(2).transpose(1, 2)  # [num_props, 49, 256]
            abnormal_feature = abnormal_feature + self.pos_embed  # 加入位置编码
            abnormal_feature = self.self_cross_attn_blocks(abnormal_feature)
            abnormal_feature = abnormal_feature.transpose(1, 2).view(-1, 256, 7, 7)
            return abnormal_feature


