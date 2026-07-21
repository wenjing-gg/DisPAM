import math
from typing import Tuple, Callable, Optional

import numpy as np
import torch
from timm.layers import (
    DropPath,
    GluMlp,
    Mlp,
    PatchDropout,
    SwiGLU,
    trunc_normal_,
    apply_keep_indices_nlc,
    apply_rot_embed_cat,
    RotaryEmbeddingCat,
    use_fused_attn,
)
from torch import nn
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint


class EvaAttention(nn.Module):
    """EVA attention block retained locally to avoid importing timm's model zoo."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        num_prefix_tokens: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
        norm_layer: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = dim // self.num_heads if attn_head_dim is None else int(attn_head_dim)
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim**-0.5
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.fused_attn = use_fused_attn()
        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer(
                    "k_bias",
                    torch.zeros(all_head_dim),
                    persistent=False,
                )
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        if self.qkv is not None:
            qkv_bias = (
                torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.q_bias is not None
                else None
            )
            qkv = torch.nn.functional.linear(x, self.qkv.weight, qkv_bias)
            qkv = qkv.reshape(
                batch,
                tokens,
                3,
                self.num_heads,
                -1,
            ).permute(2, 0, 3, 1, 4)
            query, key, value = qkv.unbind(0)
        else:
            query = self.q_proj(x).reshape(
                batch, tokens, self.num_heads, -1
            ).transpose(1, 2)
            key = self.k_proj(x).reshape(
                batch, tokens, self.num_heads, -1
            ).transpose(1, 2)
            value = self.v_proj(x).reshape(
                batch, tokens, self.num_heads, -1
            ).transpose(1, 2)
        if rope is not None:
            prefix = self.num_prefix_tokens
            query = torch.cat(
                (
                    query[:, :, :prefix],
                    apply_rot_embed_cat(query[:, :, prefix:], rope),
                ),
                dim=2,
            ).type_as(value)
            key = torch.cat(
                (
                    key[:, :, :prefix],
                    apply_rot_embed_cat(key[:, :, prefix:], rope),
                ),
                dim=2,
            ).type_as(value)
        if self.fused_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attention = (query * self.scale) @ key.transpose(-2, -1)
            if attn_mask is not None:
                attention = attention.masked_fill(
                    ~attn_mask.to(torch.bool)[:, None, None, :],
                    float("-inf"),
                )
            attention = self.attn_drop(attention.softmax(dim=-1))
            output = attention @ value
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(self.norm(output)))


class EvaBlock(nn.Module):
    """Pre-normalized EVA block used by Primus and PrimusV2."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        mlp_ratio: float = 4.0,
        swiglu_mlp: bool = False,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        num_prefix_tokens: int = 1,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        attn_head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.gamma_1 = (
            nn.Parameter(float(init_values) * torch.ones(dim))
            if init_values is not None
            else None
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp and scale_mlp:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=hidden_features,
                norm_layer=norm_layer,
                drop=proj_drop,
            )
        elif swiglu_mlp:
            self.mlp = GluMlp(
                in_features=dim,
                hidden_features=hidden_features * 2,
                act_layer=nn.SiLU,
                gate_last=False,
                drop=proj_drop,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = (
            nn.Parameter(float(init_values) * torch.ones(dim))
            if init_values is not None
            else None
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention = self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
        feed_forward = self.mlp(self.norm2(x))
        if self.gamma_1 is not None:
            attention = self.gamma_1 * attention
            feed_forward = self.gamma_2 * feed_forward
        x = x + self.drop_path1(attention)
        return x + self.drop_path2(feed_forward)


class Eva(nn.Module):
    """Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)


    """

    def __init__(
        self,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        qkv_bias: bool = True,
        qkv_fused: bool = False,
        mlp_ratio: float = 4 * 2 / 3,
        swiglu_mlp: bool = True,
        scale_mlp: bool = True,
        scale_attn_inner: bool = False,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        proj_drop_rate: float = 0.0,  # drops out things related to the projection. That is in the MLP and at the end of EVA attention
        attn_drop_rate: float = 0.0,  # drops attention, meaning connections between patches may bebroken up at random
        drop_path_rate: float = 0.0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        norm_layer: Callable = LayerNorm,
        init_values: Optional[float] = None,
        use_abs_pos_emb: bool = True,
        use_rot_pos_emb: bool = True,
        dynamic_img_size: bool = False,
        ref_feat_shape: Optional[Tuple[int, ...]] = None,  # 224/14
        num_reg_tokens: int = 0,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        block_fn=EvaBlock,
    ):
        """
        Diff to timm implementation

        - removed patch embedding, we expect embeded patches
        - removed classification token, we use features at the end
        - removed head
        - dynamic image size is not supported, but left in for future stuff
        - self.cls_token removed
        - removed postnorm block support
        """
        super().__init__()
        if rope_kwargs is None:
            rope_kwargs = {}

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        self.num_prefix_tokens = num_reg_tokens

        num_patches = np.prod(ref_feat_shape)

        self.pos_embed = (
            nn.Parameter(torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            # self.rope = VisionRotaryEmbeddingFast_Fabian3D(
            #     embed_dim // num_heads,
            #     ft_seq_len=ref_feat_shape
            # )
            if len(ref_feat_shape) == 3:
                rope_dim = round(embed_dim // num_heads / 1.5)
                assert rope_dim == embed_dim / num_heads / 1.5, "rope dim must be divsible by (num_heads * 1.5)"
                assert rope_dim % 4 == 0, "rope dim must be divisible by 4"
            else:
                rope_dim = embed_dim // num_heads
            self.rope = rope_impl(
                rope_dim, in_pixels=False, feat_shape=ref_feat_shape, ref_feat_shape=ref_feat_shape, **rope_kwargs
            )
        else:
            self.rope = None

        dpr = np.linspace(0, drop_path_rate, depth)  # stochastic depth decay rule
        block_fn = block_fn
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qkv_fused=qkv_fused,
                    mlp_ratio=mlp_ratio,
                    swiglu_mlp=swiglu_mlp,
                    scale_mlp=scale_mlp,
                    scale_attn_inner=scale_attn_inner,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    num_prefix_tokens=self.num_prefix_tokens,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {"pos_embed", "cls_token"}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )
        return matcher

    def _pos_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dynamic_img_size:
            raise NotImplementedError("dynamic_img_size is not implemented at the moment")
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                pos_embed = resample_abs_pos_embed_3d(
                    self.pos_embed,
                    (H, W),
                    num_prefix_tokens=self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed
            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        if pos_embed is not None:
            x = x + pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
            return x, rot_pos_embed, keep_indices
        else:
            return x, rot_pos_embed, None

    def forward_features(self, x):
        x, rot_pos_embed, keep_indices = self._pos_embed(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x, keep_indices

    def forward(self, x):
        x, keep_indices = self.forward_features(x)
        return x, keep_indices
