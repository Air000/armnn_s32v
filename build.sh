cd armnn
mkdir build_arm64
cd build_arm64

cmake .. -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DARMCOMPUTE_ROOT=/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/ComputeLibrary -DARMCOMPUTE_BUILD_DIR=/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/ComputeLibrary/build  -DBOOST_ROOT=/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/dependencies/boost/linux-arm64 -DTF_GENERATED_SOURCES=/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/tensorflow-master/_src4arm -DBUILD_TF_PARSER=1 -DPROTOBUF_ROOT=/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/dependencies/protobuf/_install_arm-linux/usr/local -DBUILD_TESTS=ON -DARMCOMPUTENEON=ON

make
