#!/bin/bash

# arguments:
    #-c = max posts
    #-i = population index

# argument handler
while getopts ":c:i:" opt; do
  case $opt in
    c) c="$OPTARG"
    ;;
    i) i="$OPTARG"
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

# compile and run (comment out make line if running <test.sh>)
cd ./src && make main && cd ../
cd "./EvoP2"
cd "./Pop-$i"
[ ! -d "robust" ] && mkdir robust
# ../../../src/main $c r
../../src/main $c r
echo "| Population Index: $i"
