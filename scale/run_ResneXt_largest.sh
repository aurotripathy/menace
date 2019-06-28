HSA_FORCE_FINE_GRAIN_PCIE=1 python2.7 micro_benchmarking_pytorch.py --network ResNeXt101_32C_8d  --dataparallel --device_ids=0,1,2,3 --iterations 10 --batch-size 64
HSA_FORCE_FINE_GRAIN_PCIE=1 python2.7 micro_benchmarking_pytorch.py --network ResNeXt101_32C_16d --dataparallel --device_ids=0,1,2,3 --iterations 10 --batch-size 64
HSA_FORCE_FINE_GRAIN_PCIE=1 python2.7 micro_benchmarking_pytorch.py --network ResNeXt101_32C_32d --dataparallel --device_ids=0,1,2,3 --iterations 10 --batch-size 64
HSA_FORCE_FINE_GRAIN_PCIE=1 python2.7 micro_benchmarking_pytorch.py --network ResNeXt101_32C_48d --dataparallel --device_ids=0,1,2,3 --iterations 10 --batch-size 64
