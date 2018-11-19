#!/bin/bash

# TODO
git submodule init
git submodule update --recursive
cd ./boost
git submodule init
git submodule update --recursive
./bootstrap.sh
./b2 headers