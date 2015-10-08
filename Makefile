CC = gcc
NVCC = nvcc

CCFLAGS = -Wall
NVCCFLAGS = -std=c++11 -rdc=true -fmad=false
LDFLAGS = -lm

DEBUG ?= 0

ifeq ($(DEBUG),1)
	CCFLAGS += -Og -g -pg -DSHOW_CYCLES
	NVCCFLAGS += -G -lineinfo
else
	CCFLAGS += -O3
endif

ALL_NVCCFLAGS = $(NVCCFLAGS)
ALL_NVCCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS = $(addprefix -Xlinker ,$(LDFLAGS))

VIDEO ?= 0
FRAMES ?= 0

ifeq ($(VIDEO),0)
	WIDTH = 352
	HEIGHT = 288
	INPUT_VIDEO = yuv/foreman.yuv
	OUTPUT_VIDEO = yuv/test.yuv
	REFERENCE_VIDEO = yuv/reference.yuv
else ifeq ($(VIDEO),1)
	WIDTH = 3840
	HEIGHT = 2160
	INPUT_VIDEO = /opt/cipr/foreman_4k.yuv
	OUTPUT_VIDEO = ~/yuv/test/foreman_4k.yuv
	REFERENCE_VIDEO = ~/yuv/reference/foreman_4k.yuv
endif

all: c63enc #c63dec c63pred

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(ALL_NVCCFLAGS) -c $< -o $@

c63enc: c63enc.o dsp.o tables.o io.o c63_write.o common.o me.o
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -o $@ $^
c63dec: c63dec.c dsp.o tables.o io.o common.o me.o
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -o $@ $^
c63pred: c63dec.c dsp.o tables.o io.o common.o me.o
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -DC63_PRED -o $@ $^

clean:
	rm -f *.o c63enc temp/* $(OUTPUT_VIDEO)

encode: c63enc
	./c63enc -w $(WIDTH) -h $(HEIGHT) -f $(FRAMES) -o temp/test.c63 $(INPUT_VIDEO)
decode:
	./c63dec temp/test.c63 $(OUTPUT_VIDEO)

vlc:
	vlc --rawvid-width $(WIDTH) --rawvid-height $(HEIGHT) --rawvid-chroma I420 $(OUTPUT_VIDEO)
vlc-original:
	vlc --rawvid-width $(WIDTH) --rawvid-height $(HEIGHT) --rawvid-chroma I420 $(INPUT_VIDEO)
vlc-reference:
	vlc --rawvid-width $(WIDTH) --rawvid-height $(HEIGHT) --rawvid-chroma I420 $(REFERENCE_VIDEO)

gprof:
	gprof c63enc gmon.out -b
gprof-file:
	gprof c63enc gmon.out > temp/gprof-result.txt
nvprof:
	nvprof ./c63enc -w $(WIDTH) -h $(HEIGHT) -o temp/test.c63 $(INPUT_VIDEO)

psnr:
	./tools/yuv-tools/ycbcr.py psnr $(INPUT_VIDEO) $(WIDTH) $(HEIGHT) IYUV $(OUTPUT_VIDEO)
psnr-reference:
	./tools/yuv-tools/ycbcr.py psnr $(INPUT_VIDEO) $(WIDTH) $(HEIGHT) IYUV $(REFERENCE_VIDEO)
psnr-diff:
	./tools/yuv-tools/ycbcr.py psnr $(REFERENCE_VIDEO) $(WIDTH) $(HEIGHT) IYUV $(OUTPUT_VIDEO)
cmp:
	@cmp --silent -n "$$(wc -c < $(OUTPUT_VIDEO))" $(OUTPUT_VIDEO) $(REFERENCE_VIDEO) && echo "test == reference :D" || echo "test != reference :("
	
cachegrind:
	valgrind --tool=cachegrind --branch-sim=yes --cachegrind-out-file=temp/cachegrind.out ./c63enc -w $(WIDTH) -h $(HEIGHT) -f 30 -o temp/test.c63 $(INPUT_VIDEO)
