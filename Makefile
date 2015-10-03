CC = nvcc -lineinfo -rdc=true
#NVCC = nvcc -lineinfo
CFLAGS = -O3 -pg #-Wall -pg -DSHOW_CYCLES
LDFLAGS = -lm

all: c63enc #c63dec c63pred

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@
	
%.o: %.cu
	$(CC) $< $(CFLAGS) -c -o $@

c63enc: c63enc.o dsp.o tables.o io.o c63_write.o common.o me.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63dec: c63dec.c dsp.o tables.o io.o common.o me.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63pred: c63dec.c dsp.o tables.o io.o common.o me.o
	$(CC) $^ -DC63_PRED $(CFLAGS) $(LDFLAGS) -o $@

clean:
	rm -f *.o c63enc temp/* #c63dec c63pred

encode:
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

cmp:
	@cmp --silent -n "$$(wc -c < yuv/test.yuv)" yuv/test.yuv yuv/reference.yuv && echo "test == reference :D" || echo "test != reference :("
	
