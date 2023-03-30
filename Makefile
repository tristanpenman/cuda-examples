CXXFLAGS=-std=c++14 -fmad=false --expt-relaxed-constexpr --compiler-options -Wall
LDFLAGS=
TARGET_ARCH=-arch=sm_60

TARGETS=\
	00-hello-world \
	01-cuda-hello-world \
	02-cuda-hello-world-faster \
	03-templates-device-ptr \
	04-templates-device-mem \
	05-thrust-rand-vectors

all: $(TARGETS)

%:%.cu
	nvcc $(TARGET_ARCH) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -rf $(TARGETS)
