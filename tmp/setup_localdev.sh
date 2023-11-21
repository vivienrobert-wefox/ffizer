#!/bin/bash

##
## USAGE:
##   ./setup_localdev.sh pypi_username pypi_password
##

USERNAME=${1:-$POETRY_HTTP_BASIC_WEFOX_USERNAME}
PASSWD=${2:-$POETRY_HTTP_BASIC_WEFOX_PASSWORD}
python -m venv .venv
source .venv/bin/activate

python --version
python -m pip install --upgrade pip

poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
if [ -z "$USERNAME" ] || [ -z "$PASSWD" ];
then
    echo -e "Warning: You will not use internal pypi"
else
    echo "Using internal pypi"
    poetry config http-basic.wefox $USERNAME $PASSWD
fi
poetry install --with dev
