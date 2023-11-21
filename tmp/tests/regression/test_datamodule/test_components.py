import pytest


@pytest.fixture
def fitting_data(global_datadir):
    ...


@pytest.fixture
def testing_data(global_datadir):
    ...


@pytest.fixture
def component():
    from ilil.data.components.mycomponent import MyComponent
    return MyComponent()


def test_scaler(global_datadir, scaler, fitting_data, testing_data, ndarrays_regression):
    # ndarrays_regression: https://pytest-regressions.readthedocs.io/en/latest/api.html#ndarrays-regression
    fitted_data = scaler.fit_transform(fitting_data)
    tested_data = scaler.transform(testing_data)

    ndarrays_regression.check({
        "fitted_data": fitted_data,
        "tested_data": tested_data
    })

