# SMuRF 特征提取器（基于 Swin Transformer）实现说明

目的：说明仓库中用于从影像（放射 CT/3D）和病理（2D patch / embedding）提取特征的实现细节，帮助你快速理解输入/输出、前向流程与复用封装建议。

位置
- 源文件（实现）：`SMuRF_MultiModal/models.py`
- 关键类：
  - `SwinTransformerRadiologyModel` — radiology / 3D CT 特征提取
  - `SwinTransformerPathologyModel` — pathology / 2D tiles（或 embeddings）特征提取
- 其它相关：`swintransformer.py`（SwinTransformer 实现）、`models.py` 中的 `Model`/`RModel`/`PModel` 使用方式

核心要点（快速概览）
- 两个 extractor 都基于同一 `SwinTransformer` 实例（来自 `swintransformer.py`），但选择不同的 intermediate hidden-state 进行后处理。
- Radiology extractor 处理 3D 张量：[B, C, D, H, W] → 输出 [B, out_channels]。
- Pathology extractor 接受按-tile 拼接的 2D 张量：[B_tiles, C, H, W]，需要 `batch_size` 参数把 tiles 重组回样本维度，最终输出 [batch_size, out_channels]。

详细说明

1) `SwinTransformerRadiologyModel`（用于 CT / 3D）
- 初始化关键参数（在 `models.py` 中）：
  - `in_channels`：输入通道数（项目中通常为 4）
  - `feature_size`：Swin embed dim（内部会扩展到 `feature_size * 4`）
  - `out_channels`：最终输出向量维度（常传 `args.feature_size`）
  - `patch_size`, `window_size`, `depths`, `num_heads` 等与 Swin 配置相关
- forward 流程（精要）：
  1. 调用 `self.swinViT(x_in, self.normalize)` 得到 `hidden_states_out`（多尺度列表）
  2. 选取 `hidden_states_out[2]`（中尺度）并 `rearrange` 通道顺序以便做 `LayerNorm`
  3. 再 `rearrange` 回原顺序后做 `AdaptiveAvgPool3d([1,1,1])`
  4. 用 `Conv3d(feature_size*4 -> out_channels, kernel=1)` 降维并 `flatten` 得到 `[B, out_channels]`
- 期望输入/输出形状：
  - 输入：`[B, C, D, H, W]`
  - 输出：`[B, out_channels]`

2) `SwinTransformerPathologyModel`（用于病理 patch / embedding / tiles）
- 特殊点：病理通常以 tile/patch 的形式批量处理，模型在 forward 时需要 `batch_size` 来把这些 tiles 重新分配回样本维度。
- forward 流程（精要）：
  1. 调用 `self.swinViT(x_in, self.normalize)` 得到 `hidden_states_out`
  2. 选取 `hidden_states_out[-1]`（最后一个尺度）并 `rearrange`，对 channel 维做 `LayerNorm`
  3. 将 tile 批次 reshape 成 `(batch_size, feature_size*4, n_instances, H_out, W_out)`
  4. 使用 `AdaptiveAvgPool3d` + `Conv3d` 降维并 `flatten` 得到 `[batch_size, out_channels]`
- 期望输入/输出形状：
  - 输入 tiles：`[B_tiles, C, H, W]`，其中 `B_tiles = batch_size * n_tiles_per_sample`
  - forward 还需传 `batch_size`（原始样本数）
  - 输出：`[batch_size, out_channels]`
- 注意：实现中 `in_channels` 在项目里为 192（见 `PModel`），所以传入 tile 的通道数必须匹配。

实现要点与设计约定
- 作者在实例化时通常设置 `feature_size = int(args.feature_size/2)` 作为 Swin 的 `embed_dim`，而最终 `out_channels = args.feature_size`（通过 1x1 Conv 投影得到）。
- 两种 extractor 都使用 `LayerNorm`（对 channel 做标准化）再做全局聚合（adaptive avgpool）。
- Swin 的 `window_size` / `patch_size` / `depths` / `num_heads` 会影响内部特征图的空间尺寸，确保这些超参与输入分辨率和 tile 大小一致。

常见陷阱（你需要留意）
- Pathology 的 `batch_size` 参数：若调用者传入的 `x_in` 已是形状 `[B, C, H, W]`（每个样本只有一个 tile），仍需传 `batch_size=B`；若 `x_in` 是多个 tile 的集合（例如 8 tiles per sample），需确保 `B_tiles == batch_size * n_tiles`。
- 输入通道不匹配会导致运行时错误（CT 4 通道、pathology embedding 192 通道）。
- 若调整 Swin 的 `depths/num_heads`，请确认 selected hidden-state index（radiology 使用 index 2、pathology 使用 -1）仍然是你需要的尺度，否则后处理时可能尺寸不符。
- `AdaptiveAvgPool3d` 与 `Conv3d` 的输入维度依赖于 `feature_size` 与 hidden-state 的通道结构，任意改动 `feature_size` 时注意更新对应 reshape/LayerNorm 尺寸。

示例封装（推荐）：统一 `FeatureExtractor` wrapper
- 目标：为上层使用提供统一接口，隐藏 pathology tiles 重组细节。
- 示例（伪代码）请看下面代码块（非仓库文件）：

```/dev/null/feature_extractor.py#L1-60
from torch import nn

class FeatureExtractor:
    \"\"\"统一封装 radiology / pathology 提取接口\"\"\"
    def __init__(self, radiology_model: nn.Module, pathology_model: nn.Module):
        self.radiology = radiology_model
        self.pathology = pathology_model

    def extract_radiology(self, x):
        \"\"\"输入 x: [B, C, D, H, W] -> 返回 [B, out_channels]\"\"\"
        return self.radiology(x)

    def extract_pathology(self, x_tiles, batch_size):
        \"\"\"输入 x_tiles: [B_tiles, C, H, W]，返回 [batch_size, out_channels]\"\"\"
        return self.pathology(x_tiles, batch_size)
```

快速调试示例（用随机张量做 smoke test）
- Radiology（B=1, C=4, D=32, H=64, W=64）：

```/dev/null/example_usage.py#L1-20
import torch
# 假设 radi_model 已实例化
x = torch.randn(1, 4, 32, 64, 64)
feats = radi_model(x)  # -> [1, out_channels]
print('radiology feats:', feats.shape)
```

- Pathology（batch_size=1, 每样本 8 tiles, C=192, tile 16x16）：

```/dev/null/example_usage.py#L21-40
import torch
# 假设 path_model 已实例化
tiles = torch.randn(1*8, 192, 16, 16)  # B_tiles = 8
feats_p = path_model(tiles, batch_size=1)  # -> [1, out_channels]
print('pathology feats:', feats_p.shape)
```

建议的下一步（可选）
- 将上面的 `FeatureExtractor` wrapper 写入 `SMuRF_MultiModal/feature_extractor.py` 以便上游复用。
- 添加一个小的 unit test（`tests/test_feature_extractor.py`），用随机张量校验 radiology/pathology 的输出形状。
- 在 `docs/README.md` 或 `AGENTS.md` 中加入对该文档的索引链接（便于新来者定位）。

简短结论
- 仓库里的 radiology/pathology 特征提取器已经完整实现并相互一致，主要差别在于 pathology 需要处理 tile-to-batch 的重组。封装一个小的 `FeatureExtractor` 可以显著提升可复用性与可读性。若你愿意，我可以把 wrapper 与 smoke-test 的实现写成文件并提交（需你授权 git 提交行为）。