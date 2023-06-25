#!/bin/bash

# arguments:
    #-c=circuit_size
    #-i=population_index
    #-p=permutation

# argumnt handler
while getopts ":c:i:p:" opt; do
  case $opt in
    c) c="$OPTARG"
    ;;
    i) i="$OPTARG"
    ;;
    p) p="$OPTARG"
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
cd "Evo$c""N/Evo-$i"
[ ! -d "record" ] && mkdir record
../../main $p