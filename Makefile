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
DEPS = embann.h
OBJ = embann.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

default: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)
