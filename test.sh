#!/bin/bash

# arguments
  #-c=circuit_size

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
make main
for ((i = 1; i <= 32; ++i));
do
    ./robust.sh -c$c -i$i
done