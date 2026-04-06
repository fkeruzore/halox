import numpy as np
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax_cosmo as jc
from importlib import resources
# use boolean to control the emus

class SigmaMEmulator:
    def __init__(self, weight_file = "sigma_40k_conv8.npz"):
        with resources.as_file(resources.files("halox.emus") / weight_file) as data_path:
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
        bounds = jnp.array([
            [11, 16],   # logMass
            [-0.05, 5],   # redshift
            [0.01, 0.08],   # Omega_b
            [0.085, 0.5],   # Omega_c
            [0.6, 1.],   # sigma8
            [0.4, 1.],   # h
            [0.8, 1.1],   # n_s
        ])

        mins = bounds[:, 0]
        maxs = bounds[:, 1]
        ranges = maxs-mins

        return (x - mins) / ranges

    # --- input builder ---
    def build_input(self, m:ArrayLike, z:ArrayLike, c:jc.Cosmology):
        m=jnp.atleast_1d(m)
        z=jnp.asarray(z)
        

        logM = jnp.log10(m)

        x = jnp.column_stack([
            logM,
            jnp.broadcast_to(z, logM.shape),
            jnp.broadcast_to(c.Omega_b, logM.shape),
            jnp.broadcast_to(c.Omega_c, logM.shape),
            jnp.broadcast_to(c.sigma8, logM.shape),
            jnp.broadcast_to(c.h, logM.shape),
            jnp.broadcast_to(c.n_s, logM.shape),
        ])
        return self.normalize(x)

    # --- public API ---
    def __call__(self, m: ArrayLike, z: ArrayLike, cosmo_ray: jc.Cosmology):
        x = self.build_input(m, z, cosmo_ray)
        return jnp.squeeze(10 ** self.forward(x))

def stack_cosmologies(cosmos: list[jc.Cosmology]) -> jc.Cosmology:
    return jc.Cosmology(
        Omega_b=jnp.array([c.Omega_b for c in cosmos]),
        Omega_c=jnp.array([c.Omega_c for c in cosmos]),
        sigma8=jnp.array([c.sigma8 for c in cosmos]),
        h=jnp.array([c.h for c in cosmos]),
        n_s=jnp.array([c.n_s for c in cosmos]),
        Omega_k=jnp.array([c.Omega_k for c in cosmos]),
        w0=jnp.array([c.w0 for c in cosmos]),
        wa=jnp.array([c.wa for c in cosmos]),
    )

if __name__ == "__main__":
    from halox.lss import sigma_M
    cosmo_fid = jc.Planck15()

    cosmo_high_s8 = jc.Cosmology(
        Omega_b=cosmo_fid.Omega_b,
        Omega_c=cosmo_fid.Omega_c,
        h=cosmo_fid.h,
        sigma8=0.9,
        n_s=cosmo_fid.n_s,
        Omega_k=0, w0=-1, wa=0
    )

    cosmo_low_s8 = jc.Cosmology(
        Omega_b=cosmo_fid.Omega_b,
        Omega_c=cosmo_fid.Omega_c,
        h=cosmo_fid.h,
        sigma8=0.7,
        n_s=cosmo_fid.n_s,
        Omega_k=0, w0=-1, wa=0
    )

    cosmologies = [
        (r"Planck15", cosmo_fid),
        (r"High $\sigma_8$", cosmo_high_s8),
        (r"Low $\sigma_8$", cosmo_low_s8),
    ]

    masses = jnp.logspace(9, 15.3, 256)
    zs = [0., 0.5, 1., 1.5, 2., 3., 4., 5.] #
    emu = SigmaMEmulator()
    sigmas = emu(masses, zs[0], cosmo_fid)
    sigmash = sigma_M(masses, zs[0], cosmo_fid)
    print(masses[::5], sigmas[::5], sigmash[::5])
