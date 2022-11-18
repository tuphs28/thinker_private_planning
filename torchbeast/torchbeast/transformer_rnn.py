# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Optional, Any, Union, Callable, Tuple
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
from torch.nn.functional import _mha_shape_check, _in_projection_packed, softmax, dropout, linear
from torch.nn.modules.transformer import _get_clones
import math, copy

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

   
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                concat_k: Optional[Tensor] = None,
                concat_v: Optional[Tensor] = None,) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
       
        x = src
        if self.norm_first:
            attn, k, v = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, concat_k, concat_v)
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            attn, k, v = self._sa_block(x, src_mask, src_key_padding_mask, concat_k, concat_v)
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))
        return x, k, v

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  concat_k: Optional[Tensor], concat_v: Optional[Tensor]) -> Tensor:
        x, _, k, v = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False,
                           concat_k=concat_k,
                           concat_v=concat_v)
        return self.dropout1(x), k, v

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True, concat_k: Optional[Tensor] = None,
                concat_v: Optional[Tensor] = None,) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights, k, v = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights,
                concat_k=concat_k, concat_v=concat_v)
        else:
            attn_output, attn_output_weights, k, v = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                concat_k=concat_k, concat_v=concat_v)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights, k, v
        else:
            return attn_output, attn_output_weights, k, v
    

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    concat_k: Optional[Tensor] = None,
    concat_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True
    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
            concat_k=concat_k,
            concat_v=concat_v
        )

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    if concat_k is not None:
        src_len = concat_k.shape[1]
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    
    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    
    if concat_k is not None:
        k = torch.cat([concat_k[:, 1:].transpose(0, 1).contiguous().view(-1, k.shape[0], k.shape[2]).transpose(0, 1), k], axis=1)
    if concat_v is not None:
        v = torch.cat([concat_v[:, 1:].transpose(0, 1).contiguous().view(-1, v.shape[0], v.shape[2]).transpose(0, 1), v], axis=1)
        
    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = softmax(attn_output_weights, dim=-1)    
    
    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, p=dropout_p)    
    attn_output = torch.bmm(attn_output_weights, v)    

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
    
    ret_k = k.transpose(0, 1).view(-1, bsz, num_heads, head_dim).transpose(0, 1)
    ret_v = v.transpose(0, 1).view(-1, bsz, num_heads, head_dim).transpose(0, 1)
    
    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
        
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)        
        return attn_output, attn_output_weights, ret_k, ret_v
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1) 
        return (attn_output, None, ret_k, ret_v)

class TransformerRNN(Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 mem_n = 16, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None):
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerRNN, self).__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, False, False,
                                                **factory_kwargs)        
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        self.num_layers = num_layers
        self.mem_n = mem_n
        self.d_model = d_model
        self.nhead = nhead        
        self.head_dim = self.d_model // self.nhead
        self._reset_parameters()    
        
        
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def forward(self, src: Tensor, core_state, notdone) -> Tensor:
        # Core state stored in the form of mask, (k, v), (k, v), ...
        # mask shape: (batch_size, mem_n)
        # key k and value v shape: (batch_size, mem_n, num_head, head_dim)
        
        src_mask = core_state[0][0]
        src_mask[~(notdone.bool()), :] = True
        src_mask[:, :-1] = src_mask[:, 1:].clone().detach()
        src_mask[:, -1] = False        
        new_core_state = [src_mask.unsqueeze(0)]        
        output = src.unsqueeze(0)
        
        bsz = src.shape[0]
        src_mask_ = src_mask.view(bsz, 1, 1, -1).broadcast_to(bsz, self.nhead, 1, -1).contiguous().view(bsz * self.nhead, 1, -1)
        ks = []
        vs = []

        for n, mod in enumerate(self.layers):
            output, new_k, new_v = mod(output, src_mask=src_mask_.detach(), concat_k=core_state[1][n], concat_v=core_state[2][n])
            ks.append(new_k.unsqueeze(0))
            vs.append(new_v.unsqueeze(0))
            
        output = self.norm(output)
        new_core_state.append(torch.cat(ks, dim=0))
        new_core_state.append(torch.cat(vs, dim=0))
        return output, new_core_state
    
    def init_state(self, bsz):
        core_state = (torch.ones(1, bsz, self.mem_n).bool(),
                      torch.zeros(self.num_layers, bsz, self.mem_n, self.nhead, self.head_dim),
                      torch.zeros(self.num_layers, bsz, self.mem_n, self.nhead, self.head_dim))
        return core_state

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, concat=False):
        super().__init__()
        self.concat = concat

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, step: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
            step: int Tensor, shape [seq_len, batch_size]
        """
        if not self.concat:
            x = x + self.pe[step, :]
        else:
            x = torch.cat([x, self.pe[step, :]], dim=-1)
        return x       


def add_hw(x, h, w):
    return x.unsqueeze(-1).unsqueeze(-1).broadcast_to(x.shape + (h,w))

def avg_last_ch(x):    
    last_mean = torch.mean(x[:, [-1]], dim=(-1, -2))
    last_mean = add_hw(last_mean, x.shape[-2], x.shape[-1])
    return torch.cat([x[:, :-1], last_mean], dim=1)    

class DepthSepConv(Module):
                   
    def __init__(self, h, w, in_channels, out_channels, kernel_size, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DepthSepConv, self).__init__()
        
        self.depth_conv = nn.Conv2d(in_channels=in_channels, 
                                    out_channels=in_channels, 
                                    kernel_size=kernel_size, 
                                    groups=in_channels, 
                                    padding='same', 
                                    **factory_kwargs)
        #self.bn = nn.BatchNorm2d(in_channels, **factory_kwargs)           
        self.bn = nn.LayerNorm2d([in_channels, h, w], **factory_kwargs)           
        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, **factory_kwargs)
        self.out = torch.nn.Sequential(self.depth_conv, self.bn, self.point_conv)
                   
    def forward(self, x):
        return self.out(x)        

class ConvTransformerEncoderLayer(Module):   

    def __init__(self, h, w, d_in, d_model=32, num_heads=8, dim_feedforward=32, mem_n=8, norm_first=False,
                 rpos=True, conv=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}        
        super(ConvTransformerEncoderLayer, self).__init__()
        
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.mem_n = mem_n        
        self.head_dim = self.embed_dim // num_heads
        self.norm_first = norm_first
        
        if not conv:
            self.h, self.w = 1, 1
            kernel_size = 1
        else:
            self.h, self.w = h, w
            kernel_size = 3
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        #self.proj = DepthSepConv(in_channels=self.embed_dim, out_channels=self.embed_dim*3, kernel_size=3, **factory_kwargs)
        self.proj = torch.nn.Conv2d(in_channels=d_in, out_channels=self.embed_dim*3, kernel_size=kernel_size, padding='same')        
        self.out = torch.nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=kernel_size, padding='same') 
                
        self.linear1 = torch.nn.Conv2d(in_channels=self.embed_dim, out_channels=dim_feedforward, kernel_size=kernel_size, padding='same') 
        self.linear2 = torch.nn.Conv2d(in_channels=dim_feedforward, out_channels=self.embed_dim, kernel_size=kernel_size, padding='same') 
        
        norm1_d = d_in + self.embed_dim if self.norm_first else self.embed_dim
        self.norm1 = nn.modules.normalization.LayerNorm((norm1_d, h, w), eps=1e-5, **factory_kwargs)        
        self.norm2 = nn.modules.normalization.LayerNorm((self.embed_dim, h, w), eps=1e-5, **factory_kwargs)         
        #self.norm2 = nn.BatchNorm2d(self.embed_dim)
        
        self.rpos = rpos
        if self.rpos:
            self.pos_w = torch.nn.Parameter(torch.zeros(self.mem_n, self.h*self.w*d_model))
            self.pos_b = torch.nn.Parameter(torch.zeros(self.mem_n, self.num_heads))
            torch.nn.init.xavier_uniform_(self.pos_w)
            torch.nn.init.uniform_(self.pos_b, -0.1, 0.1)                            
            
    def forward(self, input, attn_mask, concat_k, concat_v) -> Tuple[Tensor, Optional[Tensor]]:
        # input: Tensor of shape (1, B, C, H, W)
        # attn_mask: Tensor of shape (B*num_head, mem_n)
        # concat_k: Tensor of shape (B, mem_n, num_head, total_dim)
        # concat_v: Tensor of shape (B, mem_n, num_head, total_dim)
        T, B, C, H, W = input.shape
        tot_head_dim = H * W * self.embed_dim // self.num_heads
        
        self.tran_in = input        
        input = torch.flatten(input, 0, 1)  
        kqv = self.proj(input)       
        k, q, v = kqv[:, :self.embed_dim], kqv[:, self.embed_dim:2*self.embed_dim], kqv[:, -self.embed_dim:]
        self.k, self.q, self.v = k, q, v
        in_v = v     
        k, q, v = [x.contiguous().view(T, B * self.num_heads, self.head_dim, H, W).transpose(0, 1) for x in [k, q, v]]     
        k, q, v = [torch.flatten(x, start_dim=2) for x in [k, q, v]]   
        q_scaled = q / math.sqrt(q.shape[2])
        
        k_pre = concat_k[:, 1:].transpose(0, 1).contiguous().view(-1, B * self.num_heads, k.shape[2]).transpose(0, 1)
        k = torch.cat([k_pre, k], axis=1)
        
        if self.rpos:
            pos_w = (self.pos_w.unsqueeze(1).broadcast_to(self.mem_n, B, -1).contiguous().view(
                self.mem_n, B * self.num_heads, -1).transpose(0, 1))
            pos_b = (self.pos_b.unsqueeze(1).broadcast_to(self.mem_n, B, -1).contiguous().view(
                self.mem_n, B * self.num_heads).transpose(0, 1))
            k = k + pos_w
        
        v_pre = concat_v[:, 1:].transpose(0, 1).contiguous().view(-1, B * self.num_heads, v.shape[2]).transpose(0, 1)
        v = torch.cat([v_pre, v], axis=1)
        
        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask            
            attn_mask[:, :, -1] = +5
            self.attn_mask = attn_mask
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))            
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))        
        
        if self.rpos:
            attn_output_weights = attn_output_weights + pos_b.unsqueeze(1)
        
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)          
        self.attn_output_weights = attn_output_weights
        
        attn_output = torch.bmm(attn_output_weights, v)      
        
        attn_output = attn_output.transpose(0,1).contiguous().view(B, self.embed_dim, H, W)       
        
        #self.v = attn_output
        
        out = self.out(attn_output)        
        ret_k = k.transpose(0, 1).view(self.mem_n, B, self.num_heads, tot_head_dim).transpose(0, 1)
        ret_v = v.transpose(0, 1).view(self.mem_n, B, self.num_heads, tot_head_dim).transpose(0, 1)                       
        
        if self.norm_first:
            out = input[:, :self.embed_dim] + F.relu(out)
            out = out + F.relu(self.linear1(self.norm2(out)))
        else:
            out = out + input[:, :self.embed_dim]
            out = self.norm1(out)        
            out = self.norm2(out + self.linear2(F.relu(self.linear1(out))))        
        out = out.unsqueeze(0)
        return out, ret_k, ret_v
        

class ConvTransformerRNN(Module):
    def __init__(self,  d_in, h, w, d_model=32, num_heads=8, 
                 dim_feedforward=32, mem_n=8, num_layers=2, norm_first=False, 
                 rpos=True, conv=True, device=None, dtype=None):
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConvTransformerRNN, self).__init__()
        encoder_layer = ConvTransformerEncoderLayer(h=h, w=w, d_in=d_in, d_model=d_model, num_heads=num_heads, 
                                                    dim_feedforward=dim_feedforward, mem_n=mem_n, norm_first=norm_first,
                                                    rpos=rpos, conv=conv, **factory_kwargs)        
        self.layers = _get_clones(encoder_layer, num_layers)        
        self.norm =  nn.modules.normalization.LayerNorm((d_model,h,w), eps=1e-5, **factory_kwargs)
        #self.norm =  nn.BatchNorm2d(d_model)
        
        self.d_in = d_in
        self.h = h
        self.w = w
        self.d_model = d_model
        self.num_heads=num_heads
        self.dim_feedforward = dim_feedforward
        self.mem_n = mem_n
        self.num_layers = num_layers
        self.norm_first = norm_first
        
        self.head_dim = d_model // num_heads
        self.tot_head_dim = h * w * d_model // self.num_heads
        self._reset_parameters()            
        
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src: Tensor, core_state, notdone, notdone_attn=None) -> Tensor:
        # src: (batch_size, d_in, h, w)
        # Core state stored in the form of mask, (k_0, k_1, ...), (v_0, v_1, ...)
        # mask shape: (batch_size, mem_n)
        # key k and value v shape: (batch_size, mem_n, num_head, tot_head_dim)
        if notdone_attn is None: notdone_attn = notdone     
        
        bsz = src.shape[0]
        input = src.unsqueeze(0)
        
        ks, vs = [], []            
        out = core_state[3] * notdone.float().view(1, bsz, 1, 1, 1)

        src_mask = core_state[0][0]
        src_mask[~(notdone_attn.bool()), :] = True
        src_mask[:, :-1] = src_mask[:, 1:].clone().detach()
        src_mask[:, -1] = False                        
        src_mask_ = src_mask.view(bsz, 1, 1, -1).broadcast_to(bsz, self.num_heads, 1, -1).contiguous().view(bsz * self.num_heads, 1, -1)               
        new_core_state = [src_mask.unsqueeze(0)]     

        for n, mod in enumerate(self.layers):
            mod_in = torch.cat([input, out], dim=2)
            out, new_k, new_v = mod(mod_in, attn_mask=src_mask_.detach(), concat_k=core_state[1][n], concat_v=core_state[2][n])            
            ks.append(new_k.unsqueeze(0))
            vs.append(new_v.unsqueeze(0))

        new_core_state.append(torch.cat(ks, dim=0))
        new_core_state.append(torch.cat(vs, dim=0))
        new_core_state.append(out)

        core_state = new_core_state            
        out = self.norm(out[0]).unsqueeze(0)
        return out, new_core_state
    
    def init_state(self, bsz):
        core_state = (torch.ones(1, bsz, self.mem_n).bool(),
                      torch.zeros(self.num_layers, bsz, self.mem_n, self.num_heads, self.tot_head_dim),
                      torch.zeros(self.num_layers, bsz, self.mem_n, self.num_heads, self.tot_head_dim),
                      torch.zeros(1, bsz, self.d_model, self.h, self.w))
        return core_state

class ConvPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 120, concat: bool = False):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.concat = concat
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, step: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim, h, w]
            step: int Tensor, shape [seq_len, batch_size]
        """
        if not self.concat:
            x = x + self.pe[step, :].unsqueeze(-1).unsqueeze(-1)
        else:
            x = torch.concat([x, add_hw(self.pe[step, :], x.shape[-2], x.shape[-1])], dim=-3)
        return x

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size):

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels= 4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):

    def __init__(self, h, w, input_dim, hidden_dim, kernel_size, num_layers, num_steps):
        super(ConvLSTM, self).__init__()
        
        self.h = h
        self.w = w
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_steps = num_steps

        cell_list = []
        proj_list = []
        
        for i in range(0, self.num_layers):
            cell_list.append(ConvLSTMCell(input_dim=input_dim+hidden_dim*2,
                                          hidden_dim=self.hidden_dim,
                                          kernel_size=self.kernel_size,))
            proj_list.append(torch.nn.Conv2d(hidden_dim, hidden_dim, (2,1), groups=hidden_dim))

        self.cell_list = nn.ModuleList(cell_list)
        self.proj_list = nn.ModuleList(proj_list)
    
    def init_state(self, bsz):
        core_state = (torch.zeros(self.num_layers, bsz, self.hidden_dim, self.h, self.w),
                      torch.zeros(self.num_layers, bsz, self.hidden_dim, self.h, self.w))
        return core_state
    
    def forward(self, x, core_state):        
        t, b, c, h, w = x.shape
        out = core_state[0][-1]
        
        core_out = []
        for input in x:
            for _ in range(self.num_steps):
                new_core_state = ([], [])
                for n, (cell, proj) in enumerate(zip(self.cell_list, self.proj_list)):
                    cell_input = torch.concat([input, out, self.proj_max_mean(out, proj)], dim=1)
                    state = (core_state[0][n], core_state[1][n])
                    out, state =  cell(cell_input, state)
                    new_core_state[0].append(out)
                    new_core_state[1].append(state)                
                core_state = new_core_state
            core_out.append(out.unsqueeze(0))
        
        core_out = torch.cat(core_out)
        core_state = tuple(torch.cat([u.unsqueeze(0) for u in v]) for v in core_state)
                
        return core_out, core_state

    def proj_max_mean(self, out, linear_proj):
        out_mean = torch.mean(out, dim=(-1,-2), keepdim=True)
        out_max = torch.max(torch.max(out, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        proj_in = torch.cat([out_mean, out_max], dim=-2)
        out_sum = linear_proj(proj_in).broadcast_to(out.shape)
        return out_sum

class ConvAttnLSTMCell(nn.Module):

    def __init__(self, input_dims, embed_dim, kernel_size=3, num_heads=8, mem_n=8, attn=True, attn_mask_b=5):

        super(ConvAttnLSTMCell, self).__init__()
        c, h, w = input_dims

        self.input_dims = input_dims
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels= c + self.embed_dim,
                              out_channels= 5 * self.embed_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,)
        
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mem_n = mem_n
        self.head_dim = embed_dim // num_heads
        self.attn = attn
        self.attn_mask_b = attn_mask_b
        
        if self.attn:

            self.proj = torch.nn.Conv2d(in_channels=c, out_channels=self.embed_dim*3, kernel_size=kernel_size, padding='same')        
            self.out = torch.nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=kernel_size, padding='same') 
            self.norm = nn.modules.normalization.LayerNorm((embed_dim, h, w), eps=1e-5)        
            self.pos_w = torch.nn.Parameter(torch.zeros(self.mem_n, h*w*embed_dim))
            self.pos_b = torch.nn.Parameter(torch.zeros(self.mem_n, self.num_heads))        
            torch.nn.init.xavier_uniform_(self.pos_w)
            torch.nn.init.uniform_(self.pos_b, -0.1, 0.1)            

    def forward(self, input, h_cur, c_cur, concat_k, concat_v, attn_mask):
        # input: Tensor of shape (B, C, H, W)        
        # h_cur: B, embed_dim, H, W
        # c_ur:  B, embed_dim, H, W
        # concat_k: Tensor of shape (B, mem_n, num_head, total_dim)
        # concat_v: Tensor of shape (B, mem_n, num_head, total_dim)
        # attn_mask: Tensor of shape (B * num_head, 1, mem_n)

        combined = torch.cat([input, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g, cc_a = torch.split(combined_conv, self.embed_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)        
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g 
        if self.attn:
            a = torch.sigmoid(cc_a)
            attn_out, concat_k, concat_v = self.attn_output(input, attn_mask, concat_k, concat_v)
            c_next = c_next + a * torch.tanh(attn_out)
            self.a = a
        else:
            concat_k, concat_v = None, None

        h_next = o * torch.tanh(c_next)
        return h_next, c_next, concat_k, concat_v
    
    def attn_output(self, input, attn_mask, concat_k, concat_v):
        
        B, C, H, W = input.shape
        tot_head_dim = H * W * self.embed_dim // self.num_heads
        
        kqv = self.proj(input)       
        k, q, v = torch.split(kqv, self.embed_dim, dim=1) 
        k, q, v = [x.unsqueeze(0).contiguous().view(1, B * self.num_heads, self.head_dim, H, W).transpose(0, 1) for x in [k, q, v]]     
        k, q, v = [torch.flatten(x, start_dim=2) for x in [k, q, v]]   
        q_scaled = q / math.sqrt(q.shape[2])
        
        k_pre = concat_k[:, 1:].transpose(0, 1).contiguous().view(-1, B * self.num_heads, k.shape[2]).transpose(0, 1)
        k = torch.cat([k_pre, k], axis=1)
        
        
        pos_w = (self.pos_w.unsqueeze(1).broadcast_to(self.mem_n, B, -1).contiguous().view(
            self.mem_n, B * self.num_heads, -1).transpose(0, 1))
        pos_b = (self.pos_b.unsqueeze(1).broadcast_to(self.mem_n, B, -1).contiguous().view(
            self.mem_n, B * self.num_heads).transpose(0, 1))
        
        k = k + pos_w
        
        v_pre = concat_v[:, 1:].transpose(0, 1).contiguous().view(-1, B * self.num_heads, v.shape[2]).transpose(0, 1)
        v = torch.cat([v_pre, v], axis=1)
        
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask            
        attn_mask[:, :, -1] = self.attn_mask_b
        self.attn_mask = attn_mask
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))         
        attn_output_weights = attn_output_weights + pos_b.unsqueeze(1)        
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)                  
        self.attn_output_weights = attn_output_weights
        
        attn_output = torch.bmm(attn_output_weights, v)              
        attn_output = attn_output.transpose(0,1).contiguous().view(B, self.embed_dim, H, W) 
        
        out = self.out(attn_output)        
        out = out + input[:, :self.embed_dim]
        out = self.norm(out)
        
        ret_k = k.transpose(0, 1).view(self.mem_n, B, self.num_heads, tot_head_dim).transpose(0, 1)
        ret_v = v.transpose(0, 1).view(self.mem_n, B, self.num_heads, tot_head_dim).transpose(0, 1)  
        
        return out, ret_k, ret_v

class ConvAttnLSTM(nn.Module):

    def __init__(self, h, w, input_dim, hidden_dim, kernel_size, 
            num_layers, num_heads, mem_n, attn, attn_mask_b):
    
        super(ConvAttnLSTM, self).__init__()
        
        self.h = h
        self.w = w
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mem_n = mem_n
        self.attn = attn
        self.tot_head_dim = h * w * hidden_dim // num_heads

        layers = []
        proj_list = []
        
        for i in range(0, self.num_layers):
            layers.append(ConvAttnLSTMCell(input_dims=(input_dim+hidden_dim*2, self.h, self.w),
                                           embed_dim=self.hidden_dim,
                                           kernel_size=self.kernel_size,
                                           num_heads=num_heads,
                                           mem_n=mem_n,
                                           attn=attn,
                                           attn_mask_b=attn_mask_b))
            proj_list.append(torch.nn.Conv2d(hidden_dim, hidden_dim, (2,1), groups=hidden_dim))

        self.layers = nn.ModuleList(layers)
        self.proj_list = nn.ModuleList(proj_list)
    
    def init_state(self, bsz):        
        core_state = (torch.zeros(self.num_layers, bsz, self.hidden_dim, self.h, self.w),
                      torch.zeros(self.num_layers, bsz, self.hidden_dim, self.h, self.w),)
        if self.attn:
            core_state = core_state + (torch.zeros(self.num_layers, bsz, self.mem_n, self.num_heads, self.tot_head_dim),
                      torch.zeros(self.num_layers, bsz, self.mem_n, self.num_heads, self.tot_head_dim),
                      torch.ones(1, bsz, self.mem_n).bool())
        return core_state
    
    def forward(self, x, core_state, notdone, notdone_attn=None):        
        b, c, h, w = x.shape        
        out = core_state[0][-1] * notdone.float().view(b, 1, 1, 1)  
        
        if notdone_attn is None: notdone_attn = notdone        
        if self.attn:  
            src_mask = core_state[4][0]
            src_mask[~(notdone_attn.bool()), :] = True
            src_mask[:, :-1] = src_mask[:, 1:].clone().detach()
            src_mask[:, -1] = False                        
            new_src_mask = src_mask.unsqueeze(0)
            src_mask_reshape = src_mask.view(b, 1, 1, -1).broadcast_to(b, self.num_heads, 1, -1).contiguous().view(b * self.num_heads, 1, -1)               
        else:
            src_mask_reshape = None
                
        core_out = []
        new_core_state = ([], [], [], []) if self.attn else ([], [])
        for n, (cell, proj) in enumerate(zip(self.layers, self.proj_list)):
            cell_input = torch.concat([x, out, self.proj_max_mean(out, proj)], dim=1)
            h_cur = core_state[0][n] * notdone.float().view(b, 1, 1, 1)   
            c_cur = core_state[1][n] * notdone.float().view(b, 1, 1, 1)    
            concat_k_cur = core_state[2][n] if self.attn else None
            concat_v_cur = core_state[3][n] if self.attn else None

            h_next, c_next, concat_k, concat_v = cell(cell_input, h_cur, c_cur, 
                                                      concat_k_cur, concat_v_cur, src_mask_reshape)
            new_core_state[0].append(h_next)
            new_core_state[1].append(c_next)     
            if self.attn:
                new_core_state[2].append(concat_k)                
                new_core_state[3].append(concat_v)
            out = h_next
        
        core_state = tuple(torch.cat([u.unsqueeze(0) for u in v]) for v in new_core_state)
        if self.attn:
            core_state = core_state + (new_src_mask,)
        
        core_out = out.unsqueeze(0)
        return core_out, core_state

    def proj_max_mean(self, out, linear_proj):
        out_mean = torch.mean(out, dim=(-1,-2), keepdim=True)
        out_max = torch.max(torch.max(out, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        proj_in = torch.cat([out_mean, out_max], dim=-2)
        out_sum = linear_proj(proj_in).broadcast_to(out.shape)
        return out_sum