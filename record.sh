#!/bin/bash

# arguments:
    #-c = max posts
    #-i = population index
    #-p = permutation

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
cd ./src && make main && cd ../
cd "EvoP$c/Pop-$i"
[ ! -d "record" ] && mkdir record
../../src/main $c $p
