# ioio

## Dev setup (using poetry)

Ask your teammate for the pypi password

```sh
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
    brew install pyenv
    eval "$(pyenv init -)"
    pyenv install 3.10.9
    pyenv local 3.10.9
    ./setup_localdev.sh wefox-server <pypi_password>

```

## Run tests

Pytest is a dev dependency of this project.

```sh
    poetry run pytest
```

## Code formatting

Black is a dev dependency of this project.

```sh
    poetry run python -m black --check .
```

Another option is to have black formatting activated on file save with your IDE. For studio code see [here](https://code.visualstudio.com/docs/python/editing#_formatting)
