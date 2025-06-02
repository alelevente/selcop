#!/bin/bash

for seed in '42' '1234' '1867' '613' '1001' '704' '882' '405' '269' '120' ;
  do
    sumo -c ../02_scenario/hats.sumocfg --seed $seed --edgedata-output ../../02_data/01_simulation_results/edge_data_$seed.xml -a ../02_scenario/additionals/edge_meas_$seed.add.xml &
  done

