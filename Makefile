# SPDX-License-Identifier: GPL-2.0-only
#
#   Makefile - EMbedded Backpropogating Artificial Neural Network.
#   Copyright Peter Frost 2019

ifeq ($(OS),Windows_NT)
	CC=gcc
	CPP_CHECK=./tools/cppcheck/cppcheck.exe
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		CC=gcc
		CPP_CHECK=./tools/cppcheck/cppcheck-linux
	endif
	ifeq ($(UNAME_S),Darwin)
		CC=gcc-9
		CPP_CHECK=./tools/cppcheck/cppcheck-mac
	endif
endif

# Run the script to (check if we need to) generate the static variable file
FOO := $(shell python ./generate-static-var.py)

EXE = embann

SRC_DIR = src
OBJ_DIR = obj
INC_DIRS = -Iinclude

SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
LIBS = -lm

OPT_CFLAGS = -O2 -ftree-vectorize -ffast-math -march=native # -flto
CFLAGS = $(OPT_CFLAGS) -Wall -Wno-format -Wvla -fopenmp -fverbose-asm -fopt-info-all-vec=opt.log --save-temps #-masm=intel -fdump-final-insns -std=gnu99
GEN_PROFILE_CFLAGS = -fprofile-generate -fprofile-update=single
USE_PROFILE_CFLAGS = -fprofile-use
GEN_COVERAGE_CFLAGS = -fprofile-arcs -ftest-coverage
GEN_TREE_CFLAGS = -fdump-tree-optimized-graph
GRAPH_PDF_NAME = embann-graph.pdf

.PHONY: clean check debug generate-profile use-profile menuconfig all graph clean-keep-profile check-all profile generate-coverate test

all: $(EXE)



$(EXE): $(OBJ)
	$(CC) $^ $(INC_DIRS) $(LIBS) $(CFLAGS) -rdynamic -o $(EXE)
	$(info ### executable is located at ${EXE})

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(INC_DIRS) $(CFLAGS) -c $< -o $@




debug: CFLAGS += -g -pg
debug: all

profile: 
	./make-profiled.sh

generate-profile: CFLAGS += $(GEN_PROFILE_CFLAGS) $(GEN_TREE_CFLAGS)
generate-profile: all
	

use-profile: CFLAGS += $(USE_PROFILE_CFLAGS) $(GEN_TREE_CFLAGS)
use-profile: GRAPH_PDF_NAME := after-profile.pdf
use-profile: | all graph



coverage:
	./make-coverage.sh

generate-coverage: CFLAGS += $(GEN_COVERAGE_CFLAGS) 
generate-coverage: all





clean:
	rm -f ./$(OBJ_DIR)/* ./*.out ./*.s ./*.i ./*.res ./$(SRC_DIR)/*.c.dump \
	./$(SRC_DIR)/*.gcda ./$(EXE) ./$(EXE)-generate-profile \
	./opt.log ./$(SRC_DIR)/$(EXE) ./embann.ltrans0* ./embann.wpa* \
	./$(OBJ_DIR)/embann.c.* ./*.gcov

clean-keep-profile:
	rm -f ./$(OBJ_DIR)/*.o ./*.out ./*.s ./*.i ./*.res ./$(SRC_DIR)/*.c.dump \
	./$(EXE) ./$(EXE)-generate-profile ./opt.log \
	./$(SRC_DIR)/$(EXE) ./embann.ltrans0* ./embann.wpa* \
	./$(OBJ_DIR)/embann.c.* ./*.gcov

check:
	$(CPP_CHECK) --inline-suppr --max-configs=1 --addon=cert --addon=./tools/cppcheck/addons/misra.json ./ -i./cppcheck -UARDUINO

check-all:
	$(CPP_CHECK) --inline-suppr --force --addon=cert --addon=./tools/cppcheck/addons/misra.json ./ -i./cppcheck -UARDUINO

menuconfig:
	python ./tools/Kconfiglib/menuconfig.py ./Kconfig
	python ./tools/Kconfiglib/genconfig.py --header-path ./include/embann_config.h

graph: CFLAGS += $(GEN_TREE_CFLAGS)
graph: all
	dot -Tpdf $(OBJ_DIR)/embann.c.231t.optimized.dot -o $(GRAPH_PDF_NAME)

test: CFLAGS += -DTEST_BUILD
test: all