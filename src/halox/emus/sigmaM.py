import numpy as np
import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax_cosmo as jc
from importlib import resources
# use boolean to control the emus


class SigmaMEmulator:
    """Neural network emulator for :math:`\\sigma(M, z)`.

    Wraps a pre-trained neural network that emulates the RMS variance
    of density fluctuations :math:`\\sigma(M, z)` as a function of
    halo mass, redshift, and cosmological parameters.

    Parameters
    ----------
    weight_file : str, optional
        Name of the weight file to load from the package data,
        default ``"sigma_40k_conv8.npz"``.
    """

    def __init__(self, weight_file: str = "sigma_mp4.npz"):
        with resources.as_file(
            resources.files("halox.emus") / weight_file
        ) as data_path:
            raw = np.load(data_path, allow_pickle=True)
            self.mins = raw["bounds"][:, 0]
            self.ranges = raw["bounds"][:, 1] - self.mins
            weights = {k:raw[k] for k in raw.files if k != "bounds"}


        # Convert keys → clean format
        self.params = {}
        for k, v in weights.items():
            name = k.replace("('", "").replace("')", "").replace("', '", ".")
            self.params[name] = jnp.array(v)
        
        # Detect number of layers from weight keys
        self.n_layers = sum(1 for k in self.params if k.endswith(".kernel"))

    @staticmethod
    def silu(x: Array) -> Array:
        """SiLU (Sigmoid Linear Unit) activation function.

        Parameters
        ----------
        x : Array
            Input array.

        Returns
        -------
        Array
            ``x * sigmoid(x)``
        """
        return x * jax.nn.sigmoid(x)

    # --- layers ---
    @staticmethod
    def linear(x: Array, W: Array, b: Array) -> Array:
        """Linear (fully-connected) layer.

        Parameters
        ----------
        x : Array
            Input array of shape ``(..., in_features)``.
        W : Array
            Weight matrix of shape ``(in_features, out_features)``.
        b : Array
            Bias vector of shape ``(out_features,)``.

        Returns
        -------
        Array
            Output array of shape ``(..., out_features)``.
        """
        return x @ W + b  # Flax convention

    # --- forward pass ---
    def forward(self, x: Array) -> Array:
        """Forward pass through the neural network.

        Parameters
        ----------
        x : Array
            Normalized input array of shape ``(n, 7)``.

        Returns
        -------
        Array
            Log10 of :math:`\\sigma(M, z)` predictions, shape ``(n,)``.
        """
        p = self.params

        for i in range(1, self.n_layers):
            k = f"linear{i}.kernel"
            b = f"linear{i}.bias"
            x = self.silu(self.linear(x, p[k], p[b]))

        k = f"linear{self.n_layers}.kernel"
        b = f"linear{self.n_layers}.bias"
        x = self.linear(x, p[k], p[b])

        return x.squeeze(-1)

    def normalize(self, x: Array) -> Array:
        """Normalize inputs to [0, 1] using the training bounds.

        The inputs correspond to: log10(M [h-1 Msun]), z, Omega_b,
        Omega_c, sigma8, h, n_s.

        Parameters
        ----------
        x : Array
            Input array of shape ``(..., 7)``.

        Returns
        -------
        Array
            Normalized array of shape ``(..., 7)``.
        """
        # these are the bounds that emulator was trained on,
        # only change this if you are using emulator trained
        # on different bound

        return (x - self.mins) / self.ranges

    # --- input builder ---
    def build_input(
        self, m: ArrayLike, z: ArrayLike, c: jc.Cosmology
    ) -> Array:
        """Build and normalize the network input array.

        Parameters
        ----------
        m : ArrayLike
            Halo mass [h-1 Msun]
        z : ArrayLike
            Redshift
        c : jc.Cosmology
            Underlying cosmology

        Returns
        -------
        Array
            Normalized input array of shape ``(n, 7)``.
        """
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
    def __call__(
        self, m: ArrayLike, z: ArrayLike, cosmo: jc.Cosmology
    ) -> Array:
        """Evaluate :math:`\\sigma(M, z)` using the emulator.

        Parameters
        ----------
        m : ArrayLike
            Halo mass [h-1 Msun]
        z : ArrayLike
            Redshift
        cosmo : jc.Cosmology
            Underlying cosmology

        Returns
        -------
        Array
            RMS variance :math:`\\sigma(M, z)`
        """
        x = self.build_input(m, z, cosmo)
        return jnp.squeeze(10 ** self.forward(x))
