CXX ?= g++
FLAGS = -std=c++11 -O2 -Wall -Wextra -Werror -DNOODLE2_GRAPH_MAX_NODES=16
SRC_ND := $(wildcard lib/Noodle2/src/*.cpp)
.PHONY: all test clean
all: build/test_finetuned build/test_baseline build/batch_finetuned build/batch_baseline
build/test_finetuned: src/test_golden.cpp $(SRC_ND) include/peaknet_golden.h include/peaknet_model_flash.h
	@mkdir -p build
	$(CXX) $(FLAGS) -Ilib/Noodle2/src -Iinclude $(SRC_ND) $< -o $@
build/test_baseline: src/test_golden.cpp $(SRC_ND) baseline/peaknet_golden.h baseline/peaknet_model_flash.h
	@mkdir -p build
	$(CXX) $(FLAGS) -Ilib/Noodle2/src -Ibaseline $(SRC_ND) $< -o $@
build/batch_finetuned: src/batch_noodle2.cpp $(SRC_ND) include/peaknet_golden.h include/peaknet_model_flash.h
	@mkdir -p build
	$(CXX) $(FLAGS) -Ilib/Noodle2/src -Iinclude $(SRC_ND) $< -o $@
build/batch_baseline: src/batch_noodle2.cpp $(SRC_ND) baseline/peaknet_golden.h baseline/peaknet_model_flash.h
	@mkdir -p build
	$(CXX) $(FLAGS) -Ilib/Noodle2/src -Ibaseline $(SRC_ND) $< -o $@
test: build/test_finetuned build/test_baseline
	./build/test_finetuned
	./build/test_baseline
clean:
	rm -rf build
