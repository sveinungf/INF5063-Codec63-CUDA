CC = gcc
NVCC = nvcc

DEBUG ?= 0

VIDEO ?= 0
FRAMES ?= 0

CCFLAGS = -Wall
NVCCFLAGS = -std=c++11 -fmad=false -arch sm_50
PTXFLAGS = -warn-double-usage -warn-lmem-usage -warn-spills
LDFLAGS = -lm

ifeq ($(DEBUG),1)
	CCFLAGS += -Og -g -pg -DSHOW_CYCLES
	NVCCFLAGS += -G -lineinfo
else
	CCFLAGS += -O3
endif

ALL_NVCCFLAGS = $(NVCCFLAGS)
ALL_NVCCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_NVCCFLAGS += $(addprefix -Xptxas ,$(PTXFLAGS))

ALL_LDFLAGS = $(addprefix -Xlinker ,$(LDFLAGS))

ifeq ($(VIDEO),0)
	WIDTH = 352
	HEIGHT = 288
	INPUT_VIDEO = /opt/cipr/foreman.yuv
	REFERENCE_VIDEO = ~/yuv/reference/foreman.yuv
else ifeq ($(VIDEO),1)
	WIDTH = 3840
	HEIGHT = 2160
	INPUT_VIDEO = /opt/cipr/foreman_4k.yuv
	REFERENCE_VIDEO = ~/yuv/reference/foreman_4k.yuv
else ifeq ($(VIDEO),2)
	WIDTH = 1920
	HEIGHT = 1080
	INPUT_VIDEO = /opt/cipr/tractor.yuv
	REFERENCE_VIDEO = ~/yuv/reference/tractor.yuv
else ifeq ($(VIDEO),3)
	WIDTH = 4096
	HEIGHT = 1680
	INPUT_VIDEO = /opt/cipr/bagadus.yuv
endif

C63_FILE = temp/test.c63
OUTPUT_VIDEO = temp/output.yuv

all: c63enc

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(ALL_NVCCFLAGS) -c $< -o $@
	
ALL_OBJECTS = c63_write.o c63enc.o common.o dsp.o io.o me.o tables.o

c63enc: $(ALL_OBJECTS)
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -o $@ $^

clean:
	rm -f c63enc $(ALL_OBJECTS) temp/*

encode: c63enc
	./c63enc -w $(WIDTH) -h $(HEIGHT) -f $(FRAMES) -o $(C63_FILE) $(INPUT_VIDEO)
decode:
	./c63dec $(C63_FILE) $(OUTPUT_VIDEO)

vlc:
	vlc --rawvid-width $(WIDTH) --rawvid-height $(HEIGHT) --rawvid-chroma I420 $(OUTPUT_VIDEO)

gprof:
	gprof c63enc gmon.out -b
nvprof:
	nvprof ./c63enc -w $(WIDTH) -h $(HEIGHT) -f $(FRAMES) -o $(C63_FILE) $(INPUT_VIDEO)

PSNR_EXEC = ./tools/libyuv-tools-r634-linux-x86_64/bin/psnr

psnr:
	$(PSNR_EXEC) -s $(WIDTH) $(HEIGHT) -v $(INPUT_VIDEO) $(OUTPUT_VIDEO)
psnr-reference:
	$(PSNR_EXEC) -s $(WIDTH) $(HEIGHT) -v $(INPUT_VIDEO) $(REFERENCE_VIDEO)
psnr-diff:
	$(PSNR_EXEC) -s $(WIDTH) $(HEIGHT) -v $(REFERENCE_VIDEO) $(OUTPUT_VIDEO)
	
cmp:
	@cmp --silent -n "$$(wc -c < $(OUTPUT_VIDEO))" $(OUTPUT_VIDEO) $(REFERENCE_VIDEO) && echo "test == reference :D" || echo "test != reference :("
