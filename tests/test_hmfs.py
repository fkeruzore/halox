import jax
import pytest
import jax.numpy as jnp
import jax_cosmo as jc
from colossus.lss import mass_function
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
test_n_k_ints = [5000, 1000]


@pytest.mark.parametrize("delta_c", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
@pytest.mark.parametrize("n_k_int", test_n_k_ints)
def test_tinker08_f_sigma(delta_c, cosmo_name, n_k_int, return_vals=False):
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]
    cosmo_c = cc.setCosmology(cosmo_c)

    ms = test_mzs[:, 0]
    zs = test_mzs[:, 1]

    f_c = jnp.array(
        [
            mass_function.massFunction(
                ms[i],
                zs[i],
                mdef=f"{delta_c:.0f}c",
                model="tinker08",
                q_in="M",
                q_out="f",
            )
            for i in range(len(test_mzs))
        ]
    )

    tinker08_f_sigma = jax.jit(
        lambda m, z: halox.hmf.tinker08_f_sigma(
            m, z, cosmo=cosmo_j, delta_c=delta_c, n_k_int=n_k_int
        )
    )

    f_h = jnp.array(
        [tinker08_f_sigma(ms[i], zs[i]) for i in range(len(test_mzs))]
    )

    if return_vals:
        return f_h, f_c
    discrepancy = f_h / f_c - 1.0
    avg_disc = jnp.mean(discrepancy)
    max_disc = jnp.max(jnp.abs(discrepancy))
    assert max_disc < 2e-2, (
        f"Bias in f(sigma): avg={avg_disc:.3e}, max={max_disc:.3e}"
    )


@pytest.mark.parametrize("delta_c", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
@pytest.mark.parametrize("n_k_int", test_n_k_ints)
def test_tinker08_dn_dnlm(delta_c, cosmo_name, n_k_int, return_vals=False):
    cosmo_j, cosmo_c = test_cosmos[cosmo_name]
    cosmo_c = cc.setCosmology(cosmo_c)

    ms = test_mzs[:, 0]
    zs = test_mzs[:, 1]

    f_c = jnp.array(
        [
            mass_function.massFunction(
                ms[i],
                zs[i],
                mdef=f"{delta_c:.0f}c",
                model="tinker08",
                q_in="M",
                q_out="dndlnM",
            )
            for i in range(len(test_mzs))
        ]
    )

    tinker08_mass_function = jax.jit(
        lambda m, z: halox.hmf.tinker08_mass_function(
            m, z, cosmo=cosmo_j, delta_c=delta_c, n_k_int=n_k_int
        )
    )
    f_h = jnp.array(
        [tinker08_mass_function(ms[i], zs[i]) for i in range(len(test_mzs))]
    )

    if return_vals:
        return f_h, f_c
    discrepancy = f_h / f_c - 1.0
    avg_disc = jnp.mean(discrepancy)
    max_disc = jnp.max(jnp.abs(discrepancy))
    assert max_disc < 2e-2, (
        f"Bias in hmf: avg={avg_disc:.3e}, max={max_disc:.3e}"
    )


if __name__ == "__main__":
    cosmo_j, cosmo_c = test_cosmos["70_0.3"]
    cosmo_c = cc.setCosmology(cosmo_c)
    f_h, f_c = test_tinker08_dn_dnlm(200.0, "70_0.3", return_vals=True)
    print(f_h / f_c)
