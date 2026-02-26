#!/usr/bin/env bash

pushd $(dirname $0)

# clean exit function
clean_exit() {
  popd
  exit $1
}

# make sure we can execute them
chmod u+x makermt
chmod u+x consel
chmod u+x catpv

sitelh=$1
prefix=$2

./makermt --puzzle $sitelh "$prefix" || clean_exit 1
./consel "$prefix.rmt" "$prefix" --no_sort --no_bp --no_pp --no_sh || clean_exit 2

echo "rank,item,obs,au,np" > "$prefix.csv"
(./catpv "$prefix.pv" || clean_exit 3) \
  | awk 'NF' \
  | tail -n +3 \
  | awk '{
      printf "%d,%d,%s,%s,%s\n", $2, $3, $4, $5, $6
    }' >> "$prefix.csv"

clean_exit 0