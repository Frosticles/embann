ifeq ($(OS),Windows_NT)
	CC=gcc
	CPP_CHECK=./cppcheck/cppcheck.exe
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		CC=gcc
		CPP_CHECK=./cppcheck/cppcheck
	endif
	ifeq ($(UNAME_S),Darwin)
		CC=gcc-9
		CPP_CHECK=./cppcheck/cppcheck-mac
	endif
endif

# log regex filter "Analyzing loop at(.|.\n)*LOOP VECTORIZED\n"

DEPS = embann.h
OBJ = embann.o
LIBS = -lm
DEFAULT_CFLAGS = -I. -O3 -march=native -Wall -Wno-format -fopenmp -fverbose-asm -fopt-info-all-optall=opt.log --save-temps #-masm=intel -fopt-info-vec-missed -ffast-math -flto 
GEN_PROFILE_CFLAGS = -fprofile-generate -fprofile-update=single
USE_PROFILE_CFLAGS = -fprofile-use
CFLAGS = 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

embann: CFLAGS := $(DEFAULT_CFLAGS)
embann: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

embann-generate-profile: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)
	./embann-generate-profile

embann-profiled: CFLAGS := $(DEFAULT_CFLAGS) $(USE_PROFILE_CFLAGS)
embann-profiled: $(OBJ)
	$(CC) -c -o embann.o embann.c $(CFLAGS)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean check debug profile

clean:
	rm -f ./*.o ./*.out ./*.s ./*.i ./*.res ./*.c.dump ./*.gcda \
	./embann ./embann-generate-profile ./embann-profiled ./opt.log

check:
	CPP_CHECK --addon=cert --addon=./cppcheck/addons/misra.json ./ -i./cppcheck -UARDUINO

debug: CFLAGS += -g -pg
debug: embann

profile: CFLAGS := $(DEFAULT_CFLAGS) $(GEN_PROFILE_CFLAGS) 
profile: | embann-generate-profile embann-profiled