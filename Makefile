ifeq ($(OS),Windows_NT)
    CC=gcc-9
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        CC=gcc
    endif
    ifeq ($(UNAME_S),Darwin)
        CC=gcc-9
    endif
endif

CFLAGS=-I. -lm -O3 -march=native -Wall -fopenmp -fverbose-asm -ffast-math -fopt-info-all-optall=opt.log --save-temps #-masm=intel -fopt-info-vec-missed
DEPS = embann.h
OBJ = embann.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

embann: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean check

clean:
	rm -f ./*.o ./*.s ./*.i ./*.c.dump ./embann ./opt.log

check:
	./cppcheck/cppcheck --addon=cert --addon=./cppcheck/addons/misra.json ./ -i./cppcheck -UARDUINO