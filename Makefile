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

EXE = embann

SRC_DIR = src
OBJ_DIR = obj
INC_DIRS = -Iinclude

SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

LIBS = -lm
CFLAGS = -O3 -march=native -Wall -Wno-format -fopenmp -fverbose-asm -fopt-info-all-optall=opt.log --save-temps #-masm=intel -fopt-info-vec-missed -ffast-math -flto 
GEN_PROFILE_CFLAGS = -fprofile-generate -fprofile-update=single
USE_PROFILE_CFLAGS = -fprofile-use

.PHONY: clean check debug generate-profile use-profile menuconfig all

all: $(EXE)



$(EXE): $(OBJ)
	$(CC) $^ $(INC_DIRS) $(LIBS) $(CFLAGS) -o $(EXE)
	$(info ### executable is located at ${EXE})

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(INC_DIRS) $(CFLAGS) -c $< -o $@




debug: CFLAGS += -g -pg
debug: all

generate-profile: CFLAGS += $(GEN_PROFILE_CFLAGS)
generate-profile: EXE := $(SRC_DIR)/embann
generate-profile: all 
	

use-profile: CFLAGS += $(USE_PROFILE_CFLAGS)
use-profile: all



clean:
	rm -f ./$(OBJ_DIR)/* ./*.out ./*.s ./*.i ./*.res ./$(SRC_DIR)/*.c.dump \
	./$(SRC_DIR)/*.gcda ./$(EXE) ./$(EXE)-generate-profile ./$(EXE)-profiled \
	./opt.log ./$(SRC_DIR)/$(EXE)

check:
	$(CPP_CHECK) --inline-suppr --max-configs=1 --addon=cert --addon=./cppcheck/addons/misra.json ./ -i./cppcheck -UARDUINO

check-all:
	$(CPP_CHECK) --inline-suppr --force --addon=cert --addon=./cppcheck/addons/misra.json ./ -i./cppcheck -UARDUINO

menuconfig:
	python ./tools/Kconfiglib/menuconfig.py ./Kconfig
	python ./tools/Kconfiglib/genconfig.py --header-path ./include/embann_config.h