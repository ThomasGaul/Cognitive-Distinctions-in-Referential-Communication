#!/bin/bash

cd ./src && make main && cd ../
cd ./EvoP2
for ((i=3; i <= 9; ++i))
do
  cd ./Pop-$i
  ../../src/main 2 f
  cd ../
done