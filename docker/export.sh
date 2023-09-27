#!/usr/bin/env bash

./build.sh

docker save fetalheadsegalgorithm | gzip -c > FetalHeadSegAlgorithm.tar.gz
