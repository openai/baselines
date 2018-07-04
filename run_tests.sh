#!/bin/bash
python -m pytest --cov-report html --cov-report term --cov=. --rungpu
