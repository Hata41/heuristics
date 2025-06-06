from flax import linen as nn
import jax.numpy as jnp
from typing import Sequence, Tuple, Union, Optional
import jax
import chex
from tensorflow_probability.substrates.jax.distributions import Categorical
from qdax.core.neuroevolution.networks.networks import MLP
from jumanji.environments.packing.bin_pack.types import Observation
from qdax_binpack.jumanji_conversion import observation_to_arrays

class TransformerBlock(nn.Module):
    """Transformer block with post layer norm, implementing Attention Is All You Need
    [Vaswani et al., 2016].
    EXAMPLE USAGE:
        attention_kwargs = dict(
            num_heads=8,
            qkv_features=16,
            kernel_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros)
    """

    attention_kwargs: dict
    mlp_depth: int = 2

    @nn.compact
    def __call__(
        self,
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """Computes in this order:
            - (optionally masked) MHA with queries, keys & values
            - skip connection
            - layer norm
            - MLP
            - skip connection
            - layer norm

        This module broadcasts over zero or more 'batch-like' leading dimensions.

        Args:
            query: embeddings sequence used to compute queries; shape [..., T', D_q].
            key: embeddings sequence used to compute keys; shape [..., T, D_k].
            value: embeddings sequence used to compute values; shape [..., T, D_v].
            mask: optional mask applied to attention weights; shape [..., H=1, T', T].

        Returns:
            A new sequence of embeddings, consisting of a projection of the
                attention-weighted value projections; shape [..., T', D'].
        """

        # Multi-head attention and residual connection
        attn_output = nn.MultiHeadAttention(**self.attention_kwargs)(
            inputs_q=query, inputs_k=key, inputs_v=value, mask=mask
        )
        h = attn_output + query  # First residual connection
        h = nn.LayerNorm(use_scale=False)(h)

        # MLP and residual connection
        mlp_output = MLP(
            [self.attention_kwargs['qkv_features'] * 2] * self.mlp_depth
            + [self.attention_kwargs['qkv_features']]
        )(h)
        h = mlp_output + h  # Second residual connection
        out = nn.LayerNorm(use_scale=False)(h)
        return out
 
class BPSquashInput(nn.Module):
    """Flattens a Observation."""

    @nn.compact
    def __call__(self, observation) -> chex.Array:
        x = jnp.concatenate([
            observation.ems,
            observation.items,
        ], axis=-1)
        return x

class IdentityInput(nn.Module):
    """Only Observation Input."""

    @nn.compact
    def __call__(self, observation: Observation) -> Observation:
        return observation

class BinPackActor(nn.Module):
    torso: nn.Module
    action_head: nn.Module
    input_layer: nn.Module = IdentityInput()

    @nn.compact
    def __call__(self, observation: Observation) -> jnp.ndarray:
        observation = self.input_layer(observation)
        ems_embeddings, items_embeddings = self.torso(observation)
        
        action_probabilities = self.action_head(
            ems_embeddings, items_embeddings, action_mask=observation.action_mask
        )        
        return action_probabilities

# In nets.py

class BPActorHead(nn.Module):
    # action_dim is not strictly needed here if we infer from logits shape,
    # but can be kept for clarity or future use.
    # action_dim: Union[int, Sequence[int]] = None 
    
    @nn.compact
    def __call__(self, ems_embeddings, items_embeddings, action_mask) -> chex.Array: # Return type is Array, not Categorical
        # ems_embeddings: (..., E, D)
        # items_embeddings: (..., I, D)
        # action_mask: (..., E, I)
        
        logits = jnp.einsum("...ek,...ik->...ei", ems_embeddings, items_embeddings)
        # logits shape: (..., E, I)
        
        masked_logits = jnp.where(action_mask, logits, jnp.finfo(jnp.float32).min)
        
        # Flatten the last two dimensions (E, I) to treat as a single categorical choice
        original_shape = masked_logits.shape
        num_actions_flat = original_shape[-2] * original_shape[-1] # E * I
        
        # Reshape to (..., E * I)
        flattened_logits = masked_logits.reshape(*original_shape[:-2], num_actions_flat)
        
        # Apply softmax over the flattened E*I choices
        action_probabilities_flat = jax.nn.softmax(flattened_logits, axis=-1)
        
        return action_probabilities_flat # Shape: (..., E * I)


class BinPackTorso(nn.Module):
    """attention_kwargs = dict(
        num_heads=transformer_num_heads,
        qkv_features=qkv_features,
        kernel_init=nn.initializers.ones,
        bias_init=nn.initializers.zeros
        )"""
    num_transformer_layers: int
    attention_kwargs : dict
    
    @nn.compact
    def __call__(self, observation) -> Tuple[chex.Array, chex.Array]:

        # Item/EMS encoder
        ems_embeddings = nn.Dense(self.attention_kwargs['qkv_features'])(observation.ems)
        items_embeddings = nn.Dense(self.attention_kwargs['qkv_features'])(observation.items)
        
        # Item/Ems Masks
        items_mask = self._make_self_attention_mask(
            observation.items_mask & ~observation.items_placed
        )
        ems_mask = self._make_self_attention_mask(observation.ems_mask)

        # Decoder
        ems_cross_items_mask = jnp.expand_dims(observation.action_mask, axis=-3)
        items_cross_ems_mask = jnp.expand_dims(
            jnp.moveaxis(observation.action_mask, -1, -2), axis=-3
        )

        for _ in range(self.num_transformer_layers):
            # Self-attention on EMSs.
            ems_embeddings = TransformerBlock(self.attention_kwargs)(ems_embeddings, ems_embeddings, ems_embeddings, ems_mask)
            # Self-attention on items.
            items_embeddings = TransformerBlock(self.attention_kwargs)(items_embeddings, items_embeddings, items_embeddings, items_mask)
            # Cross-attention EMSs on items.
            new_ems_embeddings = TransformerBlock(self.attention_kwargs)(ems_embeddings, items_embeddings, items_embeddings, ems_cross_items_mask)
            # Cross-attention items on EMSs.
            items_embeddings = TransformerBlock(self.attention_kwargs)(items_embeddings, ems_embeddings, ems_embeddings, items_cross_ems_mask)
            ems_embeddings = new_ems_embeddings

        return ems_embeddings, items_embeddings

    def _make_self_attention_mask(self, mask: chex.Array) -> chex.Array:
        # Use the same mask for the query and the key.
        mask = jnp.einsum("...i,...j->...ij", mask, mask)
        # Expand on the head dimension.
        mask = jnp.expand_dims(mask, axis=-3)
        return mask

class Obs_to_Arrays(nn.Module):
    """Only Observation Input."""

    @nn.compact
    def __call__(self, observation: Observation) -> Observation:
        return observation_to_arrays(observation)