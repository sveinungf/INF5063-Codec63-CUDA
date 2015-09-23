CC = gcc
CFLAGS = -O3 -Wall -g
LDFLAGS = -lm

all: c63enc c63dec c63pred

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

c63enc: c63enc.o dsp.o tables.o io.o c63_write.o c63.h common.o me.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63dec: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63pred: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
	$(CC) $^ -DC63_PRED $(CFLAGS) $(LDFLAGS) -o $@

clean:
	rm -f *.o c63enc c63dec c63pred
