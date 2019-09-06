#!/bin/bash
python -m pytest --cov-config .coveragerc --cov-report html --cov-report term --cov=. -v
