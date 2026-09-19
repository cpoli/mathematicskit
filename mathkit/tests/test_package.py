"""Package-level smoke tests: mathkit imports cleanly and every declared
subpackage is actually importable and internally consistent."""

import mathkit as mk


def test_top_level_import_exposes_declared_subpackages():
    for name in mk.__all__:
        assert hasattr(mk, name), f"mathkit.{name} listed in __all__ but not importable"


def test_version_is_a_string():
    assert isinstance(mk.__version__, str)
    assert mk.__version__.count(".") == 2


def test_domain_versions_match_top_level_version():
    """Every domain subpackage's own __version__ should track the
    top-level package version (mirrors physicskit's convention)."""
    for name in mk.__all__:
        module = getattr(mk, name)
        if hasattr(module, "__version__"):
            assert module.__version__ == mk.__version__


def test_constants_module_has_expected_names():
    assert mk.constants.PI > 3.14
    assert mk.constants.DEFAULT_MAX_ITER > 0


def test_integrators_module_has_expected_callables():
    assert callable(mk.integrators.rk4_integrate)
    assert callable(mk.integrators.dopri5_integrate)
