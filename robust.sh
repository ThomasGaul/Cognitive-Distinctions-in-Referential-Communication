#!/bin/bash

# arguments:
    #-c=circuit_size
    #-i=population_index

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

# compile and run
  # comment out make line if running <test.sh>
#make main
cd "Evo$c""N/Evo-$i"
[ ! -d "robust" ] && mkdir robust
../../main r
echo "| Population Index: $i"