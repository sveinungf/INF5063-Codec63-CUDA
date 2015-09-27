CC = gcc
NVCC = nvcc

CCFLAGS = -Wall
NVCCFLAGS = -std=c++11
LDFLAGS = -lm

DEBUG ?= 0

ifeq ($(DEBUG),1)
	CCFLAGS += -Og -g -pg -DSHOW_CYCLES
	NVCCFLAGS = -G -lineinfo
else
	CCFLAGS += -O3
endif

ALL_NVCCFLAGS = $(NVCCFLAGS)
ALL_NVCCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS = $(addprefix -Xlinker ,$(LDFLAGS))

all: c63enc #c63dec c63pred

%.o: %.c
	$(CC) $(CCFLAGS) -o $@ -c $<
%.o: %.cu
	$(NVCC) $(ALL_NVCCFLAGS) -o $@ -c $<

c63enc: c63enc.o dsp.o tables.o io.o c63_write.o common.o me.o
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -o $@ $^
c63dec: c63dec.c dsp.o tables.o io.o common.o me.o
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -o $@ $^
c63pred: c63dec.c dsp.o tables.o io.o common.o me.o
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -DC63_PRED -o $@ $^

clean:
	rm -f *.o c63enc temp/* yuv/test.yuv

encode: c63enc
	./c63enc -w 352 -h 288 -o temp/test.c63 yuv/foreman.yuv
decode:
	./c63dec temp/test.c63 yuv/test.yuv

vlc:
	vlc --rawvid-width 352 --rawvid-height 288 --rawvid-chroma I420 yuv/test.yuv
vlc-original:
	vlc --rawvid-width 352 --rawvid-height 288 --rawvid-chroma I420 yuv/foreman.yuv
vlc-reference:
	vlc --rawvid-width 352 --rawvid-height 288 --rawvid-chroma I420 yuv/reference.yuv

gprof:
	gprof c63enc gmon.out -b
gprof-file:
	gprof c63enc gmon.out > temp/gprof-result.txt

psnr:
	./tools/yuv-tools/ycbcr.py psnr yuv/foreman.yuv 352 288 IYUV yuv/test.yuv
psnr-reference:
	./tools/yuv-tools/ycbcr.py psnr yuv/foreman.yuv 352 288 IYUV yuv/reference.yuv
psnr-diff:
	./tools/yuv-tools/ycbcr.py psnr yuv/reference.yuv 352 288 IYUV yuv/test.yuv
	
cachegrind:
	valgrind --tool=cachegrind --branch-sim=yes --cachegrind-out-file=temp/cachegrind.out ./c63enc -w 352 -h 288 -f 30 -o temp/test.c63 yuv/foreman.yuv

test: encode gprof
