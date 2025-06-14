import jax
import functools
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.tasks.arm import arm_scoring_function
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import default_qd_metrics

seed = 42
num_param_dimensions = 100  # num DoF arm
init_batch_size = 100
batch_size = 1024
num_iterations = 50
grid_shape = (100, 100)
min_param = 0.0
max_param = 1.0
min_descriptor = 0.0
max_descriptor = 1.0

# Init a random key
key = jax.random.key(seed)

# Init population of controllers
key, subkey = jax.random.split(key)
init_variables = jax.random.uniform(
    subkey,
    shape=(init_batch_size, num_param_dimensions),
    minval=min_param,
    maxval=max_param,
)

# Define emitter
variation_fn = functools.partial(
    isoline_variation,
    iso_sigma=0.05,
    line_sigma=0.1,
    minval=min_param,
    maxval=max_param,
)
mixing_emitter = MixingEmitter(
    mutation_fn=lambda x, y: (x, y),
    variation_fn=variation_fn,
    variation_percentage=1.0,
    batch_size=batch_size,
)

# Define a metrics function
metrics_fn = functools.partial(
    default_qd_metrics,
    qd_offset=0.0,
)

# Instantiate MAP-Elites
map_elites = MAPElites(
    scoring_function=arm_scoring_function,
    emitter=mixing_emitter,
    metrics_function=metrics_fn,
)

# Compute the centroids
centroids = compute_euclidean_centroids(
    grid_shape=grid_shape,
    minval=min_descriptor,
    maxval=max_descriptor,
)

# Initializes repertoire and emitter state
key, subkey = jax.random.split(key)
repertoire, emitter_state, metrics = map_elites.init(init_variables, centroids, subkey)

# Jit the update function for faster iterations
update_fn = jax.jit(map_elites.update)

# Run MAP-Elites loop
for i in range(num_iterations):
    key, subkey = jax.random.split(key)
    (repertoire, emitter_state, metrics,) = update_fn(
        repertoire,
        emitter_state,
        subkey,
    )

# Get contents of repertoire
repertoire.genotypes, repertoire.fitnesses, repertoire.descriptors