extern "C" {
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"
#include "c63_write.h"
#include "tables.h"
}

#include "common.h"
#include "dsp.h"
#include "me.h"

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

// Temporary buffers
int16_t *gpu_Y_16;
int16_t *gpu_U_16;
int16_t *gpu_V_16;

// Get CPU cycle count
uint64_t rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static bool read_yuv(FILE *file, struct c63_common *cm, yuv_t* image)
{
  size_t len = 0;

  /* Read Y. The size of Y is the same as the size of the image. The indices
     represents the color component (0 is Y, 1 is U, and 2 is V) */
  len += fread(image->Y, 1, width*height, file);

  /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
     because (height/2)*(width/2) = (height*width)/4. */
  len += fread(image->U, 1, (width*height)/4, file);

  /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
  len += fread(image->V, 1, (width*height)/4, file);

  if (ferror(file))
  {
    perror("ferror");
    exit(EXIT_FAILURE);
  }

  if (feof(file))
  {
    return false;
  }
  else if (len != width*height*1.5)
  {
    fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
    fprintf(stderr, "Wrong input? (height: %d width: %d)\n", height, width);

    return false;
  }

  return true;
}

static void zero_out_prediction(struct c63_common* cm)
{
	struct frame* frame = cm->curframe;
	memset(frame->predicted->Y, 0, cm->ypw * cm->yph * sizeof(uint8_t));
	memset(frame->predicted->U, 0, cm->upw * cm->uph * sizeof(uint8_t));
	memset(frame->predicted->V, 0, cm->vpw * cm->vph * sizeof(uint8_t));

	cudaMemset(frame->predicted_gpu->Y, 0, cm->ypw * cm->yph * sizeof(uint8_t));
	cudaMemset(frame->predicted_gpu->U, 0, cm->upw * cm->uph * sizeof(uint8_t));
	cudaMemset(frame->predicted_gpu->V, 0, cm->vpw * cm->vph * sizeof(uint8_t));
}

static void c63_encode_image(struct c63_common *cm, yuv_t *image, yuv_t* image_gpu)
{
	// Advance to next frame by swapping current and reference frame
	struct frame* temp = cm->refframe;
	cm->refframe = cm->curframe;
	cm->curframe = temp;

	cm->curframe->orig = image;
	cm->curframe->orig_gpu = image_gpu;

	/* Check if keyframe */
	if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
	{
		cm->curframe->keyframe = 1;
		cm->frames_since_keyframe = 0;

		fprintf(stderr, " (keyframe) ");
	}
	else { cm->curframe->keyframe = 0; }

	if (!cm->curframe->keyframe)
	{
		/* Motion Estimation */
		c63_motion_estimate(cm);

		/* Motion Compensation */
		c63_motion_compensate(cm);
	}
	else
	{
		// dct_quantize() expects zeroed out prediction buffers for key frames.
		// We zero them out here since we reuse the buffers from previous frames.
		zero_out_prediction(cm);
	}

	yuv_t* predicted = cm->curframe->predicted_gpu;

	/* DCT and Quantization */
	dct_quantize(cm->curframe->orig_gpu->Y, predicted->Y, cm->padw[Y_COMPONENT],
			cm->padh[Y_COMPONENT], gpu_Y_16, cm->curframe->residuals->Ydct,
			Y_COMPONENT);

	dct_quantize(cm->curframe->orig_gpu->U, predicted->U, cm->padw[U_COMPONENT],
			cm->padh[U_COMPONENT], gpu_U_16, cm->curframe->residuals->Udct,
			U_COMPONENT);

	dct_quantize(cm->curframe->orig_gpu->V, predicted->V, cm->padw[V_COMPONENT],
			cm->padh[V_COMPONENT], gpu_V_16, cm->curframe->residuals->Vdct,
			V_COMPONENT);

	/* Reconstruct frame for inter-prediction */
	dequantize_idct(gpu_Y_16, predicted->Y,
			cm->ypw, cm->yph, cm->curframe->recons_gpu->Y, cm->curframe->recons->Y, Y_COMPONENT);
	dequantize_idct(gpu_U_16, predicted->U,
			cm->upw, cm->uph, cm->curframe->recons_gpu->U, cm->curframe->recons->U, U_COMPONENT);
	dequantize_idct(gpu_V_16, predicted->V,
			cm->vpw, cm->vph, cm->curframe->recons_gpu->V, cm->curframe->recons->V, V_COMPONENT);

	/* Function dump_image(), found in common.c, can be used here to check if the
     prediction is correct */

	write_frame(cm);

	++cm->framenum;
	++cm->frames_since_keyframe;
}

static void set_searchrange_boundaries_cuda(c63_common* cm)
{
	int hY = cm->padh[Y_COMPONENT];
	int hUV = cm->padh[U_COMPONENT];

	int wY = cm->padw[Y_COMPONENT];
	int wUV = cm->padw[U_COMPONENT];

	int* leftsY = new int[cm->mb_colsY];
	int* leftsUV = new int[cm->mb_colsUV];
	int* rightsY = new int[cm->mb_colsY];
	int* rightsUV = new int[cm->mb_colsUV];
	int* topsY = new int[cm->mb_rowsY];
	int* topsUV = new int[cm->mb_rowsUV];
	int* bottomsY = new int[cm->mb_rowsY];
	int* bottomsUV = new int[cm->mb_rowsUV];

	for (int mb_x = 0; mb_x < cm->mb_colsY; ++mb_x) {
		leftsY[mb_x] = mb_x * 8 - ME_RANGE_Y;
		rightsY[mb_x] = mb_x * 8 + ME_RANGE_Y;

		if (leftsY[mb_x] < 0) {
			leftsY[mb_x] = 0;
		}

		if (rightsY[mb_x] > (wY - 8)) {
			rightsY[mb_x] = wY - 8;
		}
	}

	for (int mb_x = 0; mb_x < cm->mb_colsUV; ++mb_x) {
		leftsUV[mb_x] = mb_x * 8 - ME_RANGE_UV;
		rightsUV[mb_x] = mb_x * 8 + ME_RANGE_UV;

		if (leftsUV[mb_x] < 0) {
			leftsUV[mb_x] = 0;
		}

		if (rightsUV[mb_x] > (wUV - 8)) {
			rightsUV[mb_x] = wUV - 8;
		}
	}

	for (int mb_y = 0; mb_y < cm->mb_rowsY; ++mb_y) {
		topsY[mb_y] = mb_y * 8 - ME_RANGE_Y;
		bottomsY[mb_y] = mb_y * 8 + ME_RANGE_Y;

		if (topsY[mb_y] < 0) {
			topsY[mb_y] = 0;
		}

		if (bottomsY[mb_y] > (hY - 8)) {
			bottomsY[mb_y] = hY - 8;
		}
	}

	for (int mb_y = 0; mb_y < cm->mb_rowsUV; ++mb_y) {
		topsUV[mb_y] = mb_y * 8 - ME_RANGE_UV;
		bottomsUV[mb_y] = mb_y * 8 + ME_RANGE_UV;

		if (topsUV[mb_y] < 0) {
			topsUV[mb_y] = 0;
		}

		if (bottomsUV[mb_y] > (hUV - 8)) {
			bottomsUV[mb_y] = hUV - 8;
		}
	}

	cudaMemcpy(cm->cuda_me.leftsY_gpu, leftsY, cm->mb_colsY * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cm->cuda_me.leftsUV_gpu, leftsUV, cm->mb_colsUV * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cm->cuda_me.rightsY_gpu, rightsY, cm->mb_colsY * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cm->cuda_me.rightsUV_gpu, rightsUV, cm->mb_colsUV * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cm->cuda_me.topsY_gpu, topsY, cm->mb_rowsY * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cm->cuda_me.topsUV_gpu, topsUV, cm->mb_rowsUV * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cm->cuda_me.bottomsY_gpu, bottomsY, cm->mb_rowsY * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cm->cuda_me.bottomsUV_gpu, bottomsUV, cm->mb_rowsUV * sizeof(int), cudaMemcpyHostToDevice);

	delete[] leftsY;
	delete[] leftsUV;
	delete[] rightsY;
	delete[] rightsUV;
	delete[] topsY;
	delete[] topsUV;
	delete[] bottomsY;
	delete[] bottomsUV;
}

static void init_cuda_data(c63_common* cm)
{
	cuda_data_me* cuda_me = &(cm->cuda_me);

	cudaMalloc((void**) &(cuda_me->leftsY_gpu), cm->mb_colsY * sizeof(int));
	cudaMalloc((void**) &(cuda_me->leftsUV_gpu), cm->mb_colsUV * sizeof(int));
	cudaMalloc((void**) &(cuda_me->rightsY_gpu), cm->mb_colsY * sizeof(int));
	cudaMalloc((void**) &(cuda_me->rightsUV_gpu), cm->mb_colsUV * sizeof(int));
	cudaMalloc((void**) &(cuda_me->topsY_gpu), cm->mb_rowsY * sizeof(int));
	cudaMalloc((void**) &(cuda_me->topsUV_gpu), cm->mb_rowsUV * sizeof(int));
	cudaMalloc((void**) &(cuda_me->bottomsY_gpu), cm->mb_rowsY * sizeof(int));
	cudaMalloc((void**) &(cuda_me->bottomsUV_gpu), cm->mb_rowsUV * sizeof(int));

	cudaMalloc(&gpu_Y_16, cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT]*sizeof(int16_t));
	cudaMalloc(&gpu_U_16, cm->padw[U_COMPONENT]*cm->padh[U_COMPONENT]*sizeof(int16_t));
	cudaMalloc(&gpu_V_16, cm->padw[V_COMPONENT]*cm->padh[V_COMPONENT]*sizeof(int16_t));

	set_searchrange_boundaries_cuda(cm);
}

static void cleanup_cuda_data(c63_common* cm)
{
	cudaFree(cm->cuda_me.leftsY_gpu);
	cudaFree(cm->cuda_me.leftsUV_gpu);
	cudaFree(cm->cuda_me.rightsY_gpu);
	cudaFree(cm->cuda_me.rightsUV_gpu);
	cudaFree(cm->cuda_me.topsY_gpu);
	cudaFree(cm->cuda_me.topsUV_gpu);
	cudaFree(cm->cuda_me.bottomsY_gpu);
	cudaFree(cm->cuda_me.bottomsUV_gpu);

	cudaFree(gpu_Y_16);
	cudaFree(gpu_U_16);
	cudaFree(gpu_V_16);
}

static void copy_image_to_gpu(struct c63_common* cm, yuv_t* image, yuv_t* image_gpu)
{
	cudaMemcpy(image_gpu->Y, image->Y, cm->ypw * cm->yph * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(image_gpu->U, image->U, cm->upw * cm->uph * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(image_gpu->V, image->V, cm->vpw * cm->vph * sizeof(uint8_t), cudaMemcpyHostToDevice);
}

struct c63_common* init_c63_enc(int width, int height)
{
  int i;

  /* calloc() sets allocated memory to zero */
  struct c63_common *cm = (c63_common*) calloc(1, sizeof(struct c63_common));

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  cm->mb_colsY = cm->ypw / 8;
  cm->mb_rowsY = cm->yph / 8;
  cm->mb_colsUV = cm->mb_colsY / 2;
  cm->mb_rowsUV = cm->mb_rowsY / 2;

  /* Quality parameters -- Home exam deliveries should have original values,
   i.e., quantization factor should be 25, search range should be 16, and the
   keyframe interval should be 100. */
  cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
  //cm->me_search_range = 16;   // This is now defined in c63.h
  cm->keyframe_interval = 100;  // Distance between keyframes

  /* Initialize quantization tables */
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  cm->curframe = create_frame(cm);
  cm->refframe = create_frame(cm);

  init_cuda_data(cm);

  return cm;
}

void free_c63_enc(struct c63_common* cm)
{
	destroy_frame(cm->curframe);
	destroy_frame(cm->refframe);
	free(cm);
}

static void print_help()
{
  printf("Usage: ./c63enc [options] input_file\n");
  printf("Commandline options:\n");
  printf("  -h                             Height of images to compress\n");
  printf("  -w                             Width of images to compress\n");
  printf("  -o                             Output file (.c63)\n");
  printf("  [-f]                           Limit number of frames to encode\n");
  printf("\n");

  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	int c;

	if (argc == 1) { print_help(); }

	while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
	{
		switch (c)
		{
		case 'h':
			height = atoi(optarg);
			break;
		case 'w':
			width = atoi(optarg);
			break;
		case 'o':
			output_file = optarg;
			break;
		case 'f':
			limit_numframes = atoi(optarg);
			break;
		default:
			print_help();
			break;
		}
	}

	if (optind >= argc)
	{
		fprintf(stderr, "Error getting program options, try --help.\n");
		exit(EXIT_FAILURE);
	}

	outfile = fopen(output_file, "wb");

	if (outfile == NULL)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	struct c63_common *cm = init_c63_enc(width, height);
	cm->e_ctx.fp = outfile;

	input_file = argv[optind];

	if (limit_numframes) { printf("Limited to %d frames.\n", limit_numframes); }

	FILE *infile = fopen(input_file, "rb");

	if (infile == NULL)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	/* Encode input frames */
	int numframes = 0;

	# ifdef SHOW_CYCLES
	uint64_t kCycleCountTotal = 0;
	# endif

	yuv_t *image = create_image(cm);
	yuv_t *image_gpu = create_image_gpu(cm);

	while (1)
	{
		bool ok = read_yuv(infile, cm, image);

		if (!ok) { break; }

		copy_image_to_gpu(cm, image, image_gpu);

		printf("Encoding frame %d, ", numframes);

	# ifdef SHOW_CYCLES
		uint64_t cycleCountBefore = rdtsc();
		c63_encode_image(cm, image, image_gpu);
		uint64_t cycleCountAfter = rdtsc();

		uint64_t kCycleCount = (cycleCountAfter - cycleCountBefore)/1000;
		kCycleCountTotal += kCycleCount;
		printf("%" PRIu64 "k cycles, ", kCycleCount);
	# else
		c63_encode_image(cm, image, image_gpu);
	# endif

		printf("Done!\n");

		++numframes;

		if (limit_numframes && numframes >= limit_numframes) { break; }
	}

# ifdef SHOW_CYCLES
	printf("-----------\n");
	printf("Average CPU cycle count per frame: %" PRIu64 "k\n", kCycleCountTotal/numframes);
# endif

	destroy_image(image);
	destroy_image_gpu(image_gpu);
	cleanup_cuda_data(cm);

	free_c63_enc(cm);
	fclose(outfile);
	fclose(infile);

	return EXIT_SUCCESS;
}
