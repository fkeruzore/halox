import jax
import pytest
import jax.numpy as jnp
import jax_cosmo as jc
from colossus.lss import bias
import colossus.cosmology.cosmology as cc
import halox

jax.config.update("jax_enable_x64", True)

test_mzs = jnp.array(
    [
        [1e15, 0.0],
        [1e14, 0.0],
        [1e13, 0.0],
        [1e14, 1.0],
        [1e13, 1.0],
        [1e14, 2.0],
        [1e13, 0.0],
    ]
)
test_deltas = [200.0, 500.0]
test_cosmos = {
    "Planck18": [halox.cosmology.Planck18(), "planck18"],
    "70_0.3": [
        jc.Cosmology(0.25, 0.05, 0.7, 0.97, 0.8, 0.0, -1.0, 0.0),
        "70_0.3",
    ],
}
cc.addCosmology(
    cosmo_name="70_0.3",
    params=dict(
        flat=True,
        H0=70.0,
        Om0=0.3,
        Ob0=0.05,
        de_model="lambda",
        sigma8=0.8,
        ns=0.97,
    ),
)

G = halox.cosmology.G
tinker10_bias = jax.jit(halox.bias.tinker10_bias)


@pytest.mark.parametrize("delta_c", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_tinker10_bias(delta_c, cosmo_name, return_vals=False):
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]
    cosmo_c = cc.setCosmology(cosmo_c)

    ms = test_mzs[:, 0]
    zs = test_mzs[:, 1]

    b_c = jnp.array(
        [
            bias.haloBias(
                ms[i],
                zs[i],
                mdef=f"{delta_c:.0f}c",
                model="tinker10",
            )
            for i in range(len(test_mzs))
        ]
    )
    b_h = jnp.array(
        [
            tinker10_bias(ms[i], zs[i], cosmo=cosmo_j, delta_c=delta_c)
            for i in range(len(test_mzs))
        ]
    )

    if return_vals:
        return b_h, b_c
    discrepancy = b_h / b_c - 1.0
    avg_disc = jnp.mean(discrepancy)
    max_disc = jnp.max(jnp.abs(discrepancy))
    assert max_disc < 5e-3, (
        f"Bias in halo bias: avg={avg_disc:.3e}, max={max_disc:.3e}"
    )


if __name__ == "__main__":
    cosmo_j, cosmo_c = test_cosmos["70_0.3"]
    cosmo_c = cc.setCosmology(cosmo_c)
    b_h, b_c = test_tinker10_bias(200.0, "70_0.3", return_vals=True)
    print(b_h / b_c)
