"""
Contains model architectures for both finite width networks and NTK computations
using neural_tangents and stax.
"""
from typing import Tuple, Callable, List, Dict
import jax
import jax.numpy as jnp
from neural_tangents import stax

def CNN(
    width: int,
    num_classes: int = 10,
    mode: str = 'finite'
) -> Tuple[Callable, Callable]:
    """Creates a CNN architecture that can be used for both finite and NTK training.
    
    Args:
        width: Width factor for the network (number of channels)
        num_classes: Number of output classes
        mode: Either 'finite' for regular network or 'ntk' for NTK computation
        
    Returns:
        init_fn: Function to initialize network parameters
        apply_fn: Function to apply network to inputs
    """
    layers = [
        stax.Conv(width, (3, 3), padding='SAME'),
        stax.Relu(),
        stax.Conv(width, (3, 3), padding='SAME'),
        stax.Relu(),
        stax.AvgPool((2, 2), strides=(2, 2)),
        
        stax.Conv(width * 2, (3, 3), padding='SAME'),
        stax.Relu(),
        stax.Conv(width * 2, (3, 3), padding='SAME'),
        stax.Relu(),
        stax.AvgPool((2, 2), strides=(2, 2)),
        
        stax.Flatten(),
        stax.Dense(width * 4),
        stax.Relu(),
        stax.Dense(num_classes)
    ]
    
    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    
    if mode == 'ntk':
        return init_fn, kernel_fn
    else:
        return init_fn, apply_fn

def create_model(
    input_shape: Tuple[int, ...],
    width: int = 32,
    num_classes: int = 10,
    mode: str = 'finite',
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)
) -> Tuple[Callable, dict]:
    """Creates and initializes the model.
    
    Args:
        input_shape: Shape of input tensors (excluding batch dimension)
        width: Width factor for the network
        num_classes: Number of output classes
        mode: Either 'finite' for regular network or 'ntk' for NTK computation
        key: JAX PRNG key
        
    Returns:
        apply_fn: Function to apply network to inputs (or kernel_fn for NTK)
        params: Network parameters (None for NTK mode)
    """
    init_fn, apply_fn = CNN(width, num_classes, mode)
    
    if mode == 'finite':
        _, params = init_fn(key, (1,) + input_shape)
        return apply_fn, params
    else:
        return apply_fn, None 

def get_parameter_count(params) -> int:
    """Count total parameters in the network."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def create_models_at_different_scales(
    input_shape: Tuple[int, ...],
    width_multipliers: List[float],
    base_width: int = 32,
    num_classes: int = 10
) -> List[Tuple[Callable, Dict, Callable, int]]:
    """Create models at different scales.
    
    Args:
        width_multipliers: List of multipliers for the base width
        base_width: Base width to multiply
        
    Returns:
        List of (finite_fn, params, kernel_fn, param_count) tuples
    """
    models = []
    
    for multiplier in width_multipliers:
        width = int(base_width * multiplier)
        
        # Finite network
        apply_fn_finite, params = create_model(
            input_shape=input_shape,
            width=width,
            num_classes=num_classes,
            mode='finite'
        )
        
        # NTK
        kernel_fn, _ = create_model(
            input_shape=input_shape,
            width=width,
            num_classes=num_classes,
            mode='ntk'
        )
        
        param_count = get_parameter_count(params)
        models.append((apply_fn_finite, params, kernel_fn, param_count))
    
    return models