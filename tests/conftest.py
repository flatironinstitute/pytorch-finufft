import pytest
import torch


# custom: skip tests with 'cuda' in name when unavailable
def pytest_collection_modifyitems(session, config, items):
    if not torch.cuda.is_available():
        for test in items:
            if "cuda" in test.name:
                test.add_marker(pytest.mark.skip("CUDA unavailable"))


# When we want to explicitly run cuda/fail when cuda is missing,
# we need to disable skipping.
# From: https://github.com/jankatins/pytest-error-for-skips
# See: https://github.com/pytest-dev/pytest/issues/1364
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()

    if item.config.getoption("--error-for-skips"):
        if rep.skipped and call.excinfo.errisinstance(pytest.skip.Exception):
            rep.outcome = "failed"
            r = call.excinfo._getreprcrash()
            rep.longrepr = "Forbidden skipped test - {message}".format(
                message=r.message
            )


def pytest_addoption(parser):
    parser.addoption(
        "--error-for-skips",
        action="store_true",
        default=False,
        help="Treat skipped tests as errors",
    )
