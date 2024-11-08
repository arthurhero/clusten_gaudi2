echo $GC_KERNEL_PATH
export GC_KERNEL_PATH=/src/clusten_gaudi2/kernel/build/src/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
echo $GC_KERNEL_PATH
rm -rf build/
rm -rf kernel/build/
cd kernel
mkdir build && cd build
cmake ..
make
cd ../../
python setup.py build
