import math

import numpy as np
import torch
import torch.nn.functional as F
from tools.utils import extract, gen_coefficients
from torch import nn

from models.embedder import get_embedder


class CAPUDFNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        multires=0,
        bias=0.5,
        scale=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,
    ):
        super().__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        # [3,256,256,256,256,256,256,256,256,1]

        self.embed_fn_fine = None
        self.pe_t = TimeStepEncoding()
        self.timestep_emb = nn.Sequential(nn.Sigmoid(), nn.Linear(20, 2))
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            # embed_fn是特征转高频模块,multires表示转的高频模块的频率上线,最高sin4x则multires=4,输出的特征为[x,sinx,cosx,sin2x,cos2x,sin3x,cos3x,sin4x,cos4x],维度一共为3*(1+2*multires)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        # len(dims) = 10
        self.skip_in = skip_in
        self.scale = scale
        # dims[0]=3,这设置9层
        for l in range(self.num_layers - 1):  # noqa: E741
            if l + 1 in self.skip_in:
                # 如果下一层要进行跳跃连接则在该层的输出要减少一点以保证维度一致
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        # self.activation = nn.Softplus(beta=100)
        self.activation = nn.ReLU()

        self.act_last = nn.Sigmoid()

    def forward(self, query, time):
        query = query * self.scale
        if self.embed_fn_fine is not None:
            query = self.embed_fn_fine(query)
        q_feature = query
        timestep_emb = self.pe_t(time)
        # time:[B, Q,1],timestep_emb: [B, Q,20]
        emb_out = self.timestep_emb(timestep_emb)
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        q_feature = q_feature * (1 + scale) + shift

        for l in range(self.num_layers - 1):  # noqa: E741
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                q_feature = torch.cat([q_feature, query], -1) / np.sqrt(2)
            # 保证整体方差一致
            q_feature = lin(q_feature)

            if l < self.num_layers - 2:
                q_feature = self.activation(q_feature)

        # x = self.act_last(x)
        res = torch.abs(q_feature)
        # res = 1 - torch.exp(-x)
        return res / self.scale

    def res_udf(self, query, time):
        return self.forward(query, time)

    def udf_hidden_appearance(self, query, time):
        return self.forward(query, time)

    def gradient(self, x, time):
        """
        计算 self.udf(x, time) 对 x 的偏导数。

        参数:
            x (torch.Tensor): 需要计算梯度的输入张量。
            time (torch.Tensor or float): 时间参数。

        返回:
            torch.Tensor: self.udf(x, time) 对 x 的梯度。
        """
        x.requires_grad_(True)
        y = self.res_udf(x, time)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return gradients

    def predict(self, x, time_sum, metric="increased"):
        """
        进行多步移动
        """
        current_points = x
        time = time_sum - 1  # 假设我们从时间步10开始
        alphas = gen_coefficients(time_sum, schedule=metric).to(device=current_points.device)
        # 在evaluate阶段是的
        while time >= 0:
            # 明确require_grad以确保可以在后面求gradient
            current_points.requires_grad = True
            time_list = torch.full(
                (*current_points.shape[:-1], 1),
                time,
                dtype=torch.int64,
                device=current_points.device,
            )
            gradients_sample = self.gradient(current_points, time_list)  # 5000x3
            udf_sample = self.res_udf(current_points, time_list)  # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=-1)  # 5000x3
            coef = extract(alphas, time_list)  # 5000x1
            p_next = current_points - grad_norm * udf_sample * coef
            # 更新当前点
            current_points = p_next.detach()
            time -= 1

        udf = torch.norm(x - current_points, dim=-1, keepdim=True)
        gradient = (x - current_points) / udf
        return udf, gradient, current_points


class TimeStepEncoding:
    """
    位置编码类,用于将输入的位置向量进行频率编码。
    方法:
      __init__():
        初始化位置编码类,设置频率带的范围。
      __call__(p):
        对输入的张量 `p` 进行位置编码。
        如果输入为一维张量,会先扩展维度。
        输入的值会被归一化到 [-1, 1] 范围。
        对每个频率带,算正弦和余弦值,并将结果拼接成新的张量。
    参数:
      无显式参数。
    属性:
      freq_bands (numpy.ndarray): 频率带数组,包含 2^(0) 到 2^(L-1) 的频率值,乘以 π。
    示例:
      >>> encoder = positional_encoding_t()
      >>> p = torch.tensor([1.0, 2.0, 3.0])
      >>> encoded_p = encoder(p)
      >>> print(encoded_p)
    """

    def __init__(self):
        L = 10
        freq_bands = 2 ** (np.linspace(0, L - 1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if p.dim() == 1:
            p = p.unsqueeze(-1)
        out = []
        p = (p - 5) / 5.0
        for freq in self.freq_bands:
            out.append(torch.sin(freq * p))
            out.append(torch.cos(freq * p))
        p = torch.cat(out, dim=-1)
        return p
