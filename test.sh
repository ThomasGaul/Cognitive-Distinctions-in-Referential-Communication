#!/bin/bash

# arguments
  #-c = max posts

# argument handler
while getopts ":c:" opt; do
  case $opt in
    c) c="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

# compile and run
cd ./src && make main && cd ../
# directory to batch
cd ./OldData/EvoP2_6
for ((i = 1; i <= 32; ++i));
do
    ../../robust.sh -c$c -i$i
done
