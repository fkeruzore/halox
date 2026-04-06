import numpy as np
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax_cosmo as jc
from importlib import resources
# use boolean to control the emus


class SigmaMEmulator:
    def __init__(self, weight_file="sigma_40k_conv8.npz"):
        with resources.as_file(
            resources.files("halox.emus") / weight_file
        ) as data_path:
            raw_weights = dict(np.load(data_path, allow_pickle=True))

        # Convert keys → clean format
        self.params = {}
        for k, v in raw_weights.items():
            name = k.replace("('", "").replace("')", "").replace("', '", ".")
            self.params[name] = jnp.array(v)

    @staticmethod
    def silu(x):
        return x * jax.nn.sigmoid(x)

    # --- layers ---
    @staticmethod
    def linear(x, W, b):
        return x @ W + b  # Flax convention

    # --- forward pass ---
    def forward(self, x):
        p = self.params

        x = self.silu(self.linear(x, p["linear1.kernel"], p["linear1.bias"]))
        x = self.silu(self.linear(x, p["linear2.kernel"], p["linear2.bias"]))
        x = self.silu(self.linear(x, p["linear3.kernel"], p["linear3.bias"]))
        x = self.linear(x, p["linear4.kernel"], p["linear4.bias"])

        return x.squeeze(-1)

    @staticmethod
    def normalize(x):
        # these are the bounds that emulator was trained on,
        # only change this if you are using emulator trained
        # on different bound
        bounds = jnp.array(
            [
                [11, 16],  # logMass
                [-0.05, 5],  # redshift
                [0.01, 0.08],  # Omega_b
                [0.085, 0.5],  # Omega_c
                [0.6, 1.0],  # sigma8
                [0.4, 1.0],  # h
                [0.8, 1.1],  # n_s
            ]
        )

        mins = bounds[:, 0]
        maxs = bounds[:, 1]
        ranges = maxs - mins

        return (x - mins) / ranges

    # --- input builder ---
    def build_input(self, m: ArrayLike, z: ArrayLike, c: jc.Cosmology):
        m = jnp.atleast_1d(m)
        z = jnp.asarray(z)

        logM = jnp.log10(m)

        x = jnp.column_stack(
            [
                logM,
                jnp.broadcast_to(z, logM.shape),
                jnp.broadcast_to(c.Omega_b, logM.shape),
                jnp.broadcast_to(c.Omega_c, logM.shape),
                jnp.broadcast_to(c.sigma8, logM.shape),
                jnp.broadcast_to(c.h, logM.shape),
                jnp.broadcast_to(c.n_s, logM.shape),
            ]
        )
        return self.normalize(x)

    # --- public API ---
    def __call__(self, m: ArrayLike, z: ArrayLike, cosmo: jc.Cosmology):
        x = self.build_input(m, z, cosmo)
        return jnp.squeeze(10 ** self.forward(x))
