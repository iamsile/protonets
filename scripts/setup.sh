#!/bin/bash

# virtual environemnt directory name
ENV=".env"
# python version
# python3 --> 3.6 | python2 --> 2.7
PYTHON="python3"
# pip version
# pip3 --> python3 | pip2 --> python2
PIP="pip3"

# remove virtual environment
source scripts/remove.sh
if [ ${PYTHON} == "python2" ]; then \
    # install `virtualenv` to global pip
    ${PIP} install virtualenv
    # create virtual environment
    virtualenv -p ${PYTHON} ${ENV}
else
    # create virtual environment
    ${PYTHON} -m venv ${ENV}
fi
# activate virtual environment
source "./${ENV}/bin/activate"
# upgrade pip
pip install --upgrade pip
# run setup.py
pip install -e .[dev]
# run tests
source scripts/test.sh