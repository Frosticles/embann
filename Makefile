CC=gcc
CFLAGS=-I. 
DEPS = embann.h
OBJ = embann.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

default: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)
