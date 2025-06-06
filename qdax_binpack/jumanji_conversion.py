
import chex
from typing import NamedTuple
import jax.numpy as jnp
from jumanji.environments.packing.bin_pack.types import Observation, EMS, Item
from jumanji.environments.packing.bin_pack.space import Space

# Define the new NamedTuple for the function's output
class ArrayObservation(NamedTuple):
    """
    An observation structure where all fields are JAX arrays.
    Complex objects like EMS/Space and Item from the original Observation
    are converted into single JAX arrays by stacking their numerical components.
    The field names are identical to the original Observation NamedTuple.
    """
    ems: chex.Array         # e.g., shape (obs_num_ems, 6) or (6,)
    ems_mask: chex.Array    # e.g., shape (obs_num_ems,)
    items: chex.Array       # e.g., shape (max_num_items, 3) or (3,)
    items_mask: chex.Array  # e.g., shape (max_num_items,)
    items_placed: chex.Array# e.g., shape (max_num_items,)
    action_mask: chex.Array # e.g., shape (obs_num_ems, max_num_items)


def observation_to_arrays(obs: Observation) -> ArrayObservation:
    """
    Converts an Observation object (containing potentially nested structures
    like EMS/Space and Item) into an ArrayObservation object where all fields,
    including the representations of EMS and Item, are single JAX arrays.

    - `obs.ems` (an EMS/Space object) is transformed into a JAX array by
      stacking its 6 numerical components (x1, x2, y1, y2, z1, z2) in that order.
    - `obs.items` (an Item object) is transformed into a JAX array by
      stacking its 3 numerical components (x_len, y_len, z_len) in that order.
    - All other fields (originally masks) are ensured to be JAX arrays.

    Args:
        obs: The input Observation object.

    Returns:
        An ArrayObservation object where all fields are JAX arrays.
    """

    # Process obs.ems (which is an EMS, an alias for the Space dataclass)
    # Components are x1, x2, y1, y2, z1, z2.
    # We explicitly list them to ensure a consistent order in the stacked array.
    ems_components_list = [
        jnp.asarray(obs.ems.x1),
        jnp.asarray(obs.ems.x2),
        jnp.asarray(obs.ems.y1),
        jnp.asarray(obs.ems.y2),
        jnp.asarray(obs.ems.z1),
        jnp.asarray(obs.ems.z2),
    ]
    # Stack along the last axis.
    # If input components are 1D (shape (N,)), output is (N, 6).
    # If input components are scalars (shape ()), output is (6,).
    ems_as_array = jnp.stack(ems_components_list, axis=-1)

    # Process obs.items (which is an Item NamedTuple)
    # Components are x_len, y_len, z_len.
    items_components_list = [
        jnp.asarray(obs.items.x_len),
        jnp.asarray(obs.items.y_len),
        jnp.asarray(obs.items.z_len),
    ]
    # Stack along the last axis, similar to EMS.
    # If input components are 1D (shape (M,)), output is (M, 3).
    # If input components are scalars (shape ()), output is (3,).
    items_as_array = jnp.stack(items_components_list, axis=-1)

    # Ensure mask fields are JAX arrays
    # If they are already JAX arrays, jnp.asarray is often a no-op.
    ems_mask_as_array = jnp.asarray(obs.ems_mask)
    items_mask_as_array = jnp.asarray(obs.items_mask)
    items_placed_as_array = jnp.asarray(obs.items_placed)
    action_mask_as_array = jnp.asarray(obs.action_mask)

    # Construct and return the new ArrayObservation object
    return ArrayObservation(
        ems=ems_as_array,
        ems_mask=ems_mask_as_array,
        items=items_as_array,
        items_mask=items_mask_as_array,
        items_placed=items_placed_as_array,
        action_mask=action_mask_as_array,
    )