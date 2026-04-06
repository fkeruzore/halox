import jax
import pytest
import jax.numpy as jnp
import jax_cosmo as jc
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

test_n_k_ints = [5000, 1000]

@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
@pytest.mark.parametrize("n_k_int", test_n_k_ints)
def test_sigmaM_emu(cosmo_name, n_k_int, return_vals=False):
    cosmo_j = test_cosmos[cosmo_name][0]

    ms = test_mzs[:, 0]
    zs = test_mzs[:, 1]

    tinker08_f_sigma = jax.jit(
        lambda m, z: halox.lss.sigma_M(
            m, z, cosmo=cosmo_j, n_k_int=n_k_int
        )
    )
    f_c = jnp.array(
        [
            tinker08_f_sigma(ms[i], zs[i]).squeeze() 
            for i in range(len(test_mzs))
        ]
    )

    emu = halox.emus.SigmaMEmulator()
    tinker08_f_sigma_emu = jax.jit(
        lambda m, z: emu(
            m, z, cosmo_ray=cosmo_j
        )
    )

    f_h = jnp.array(
        [
            tinker08_f_sigma_emu(ms[i], zs[i]).squeeze() 
            for i in range(len(test_mzs))
        ]
    )

    if return_vals:
        return f_h, f_c
    discrepancy = f_c / f_h - 1.0
    avg_disc = jnp.mean(discrepancy)
    max_disc = jnp.max(jnp.abs(discrepancy))
    assert max_disc < 5e-3, (
        f"Bias in sigmaM emulator: avg={avg_disc:.3e}, max={max_disc:.3e})"
    )

@pytest.mark.parametrize("delta_c", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_tinker08_f_sigma_emu(delta_c, cosmo_name, return_vals=False):
    cosmo_j = test_cosmos[cosmo_name][0]

    ms = test_mzs[:, 0]
    zs = test_mzs[:, 1]

    tinker08_f_sigma = jax.jit(
        lambda m, z: halox.hmf.tinker08_f_sigma(
            m,
            z,
            cosmo=cosmo_j, 
            delta_c = delta_c, 
            emulate = False
            )
    )
    f_c = jnp.array(
        [tinker08_f_sigma(ms[i], zs[i]) for i in range(len(test_mzs))]
    )
    #testing on the default emulator
    tinker08_f_sigma_emu = jax.jit(
        lambda m, z: halox.hmf.tinker08_f_sigma(
            m, z, cosmo=cosmo_j, delta_c=delta_c, emulate = True
        )
    )

    f_h = jnp.array(
        [tinker08_f_sigma_emu(ms[i], zs[i]) for i in range(len(test_mzs))]
    )

    if return_vals:
        return f_h, f_c
    discrepancy = f_h / f_c - 1.0
    avg_disc = jnp.mean(discrepancy)
    max_disc = jnp.max(jnp.abs(discrepancy))
    assert max_disc < 2e-2, (
        f"Bias in f(sigma)_emu: avg={avg_disc:.3e}, max={max_disc:.3e}"
    )

@pytest.mark.parametrize("delta_c", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_tinker08_dn_dnlm_emu(delta_c, cosmo_name, return_vals=False):
    cosmo_j = test_cosmos[cosmo_name][0]

    ms = test_mzs[:, 0]
    zs = test_mzs[:, 1]

    tinker08_mass_function = jax.jit(
        lambda m, z: halox.hmf.tinker08_mass_function(
            m, z, cosmo=cosmo_j, delta_c=delta_c, emulate = False
        )
    )
    f_c = jnp.array(
        [
            tinker08_mass_function(ms[i], zs[i]) 
            for i in range(len(test_mzs))
        ]
    )
    # again, testing the default photo
    tinker08_mass_function_emu = jax.jit(
        lambda m, z: halox.hmf.tinker08_mass_function(
            m, z, cosmo=cosmo_j, delta_c=delta_c, emulate = True
        )
    )
    f_h = jnp.array(
        [
            tinker08_mass_function_emu(ms[i], zs[i])
            for i in range(len(test_mzs))
        ]
    )

    if return_vals:
        return f_h, f_c
    discrepancy = (f_h -f_c)/ f_c
    avg_disc = jnp.mean(discrepancy)
    max_disc = jnp.max(jnp.abs(discrepancy))
    assert max_disc < 2e-2, (
        f"Bias in hmf_emu: avg={avg_disc:.3e}, max={max_disc:.3e}"
    )


tinker10_bias = jax.jit(halox.bias.tinker10_bias)
@pytest.mark.parametrize("delta_c", test_deltas)
@pytest.mark.parametrize("cosmo_name", test_cosmos.keys())
def test_tinker10_bias(delta_c, cosmo_name, return_vals=False):
    cosmo_j = test_cosmos[cosmo_name][0]

    ms = test_mzs[:, 0]
    zs = test_mzs[:, 1]

    # using default emulator
    b_emu = jnp.array(
    [
        halox.bias.tinker10_bias(
            ms[i], 
            zs[i], 
            cosmo=cosmo_j, 
            delta_c=delta_c, 
            emulate = True
            )
        for i in range(len(test_mzs))]
    )
    b_h = jnp.array(
        [
            tinker10_bias(
                ms[i], 
                zs[i], 
                cosmo=cosmo_j, 
                delta_c=delta_c, 
                emulate = False
                )
            for i in range(len(test_mzs))
        ]
    )

    if return_vals:
        return b_h, b_emu
    discrepancy = b_emu / b_h - 1.0
    avg_disc = jnp.mean(discrepancy)
    max_disc = jnp.max(jnp.abs(discrepancy))
    assert max_disc < 5e-3, (
        f"Bias in halo bias_emu: avg={avg_disc:.3e}, max={max_disc:.3e}"
    )