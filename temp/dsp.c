#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "dsp.h"
#include "tables.h"

#include <immintrin.h>
#include <xmmintrin.h>

static float test1[8] __attribute__((aligned(32))) = {0.5f, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2};
static float test2[8] __attribute__((aligned(32))) = {ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

static void transpose_block(float *in_data, float *out_data)
{
    int i, j;

    __m128 row1[2], row2[2], row3[2], row4[2];
    
    float out_data2[64];
    int k;
    for(i = 0; i < 8; i +=4 )
    {
		j = k = 0;
		
		row1[k] = _mm_load_ps(&in_data[(j*8)+i]);
		row2[k] = _mm_load_ps(&in_data[((j+1)*8)+i]);
		row3[k] = _mm_load_ps(&in_data[((j+2)*8)+i]);
		row4[k] = _mm_load_ps(&in_data[((j+3)*8)+i]);
		_MM_TRANSPOSE4_PS(row1[k], row2[k], row3[k], row4[k]);

		j += 4;
		++k;
					
		row1[k] = _mm_load_ps(&in_data[(j*8)+i]);
		row2[k] = _mm_load_ps(&in_data[((j+1)*8)+i]);
		row3[k] = _mm_load_ps(&in_data[((j+2)*8)+i]);
		row4[k] = _mm_load_ps(&in_data[((j+3)*8)+i]);
		_MM_TRANSPOSE4_PS(row1[k], row2[k], row3[k], row4[k]);
		
		_mm256_store_ps(&out_data[(i*8)], _mm256_insertf128_ps(_mm256_castps128_ps256(row1[0]), row1[1], 0b00000001));
		_mm256_store_ps(&out_data[((i+1)*8)], _mm256_insertf128_ps(_mm256_castps128_ps256(row2[0]), row2[1], 0b00000001));
		_mm256_store_ps(&out_data[((i+2)*8)], _mm256_insertf128_ps(_mm256_castps128_ps256(row3[0]), row3[1], 0b00000001));
		_mm256_store_ps(&out_data[((i+3)*8)], _mm256_insertf128_ps(_mm256_castps128_ps256(row4[0]), row4[1], 0b00000001));
		// Better to just store 128 bit at a time?	
	}
}

static void dct_1d(float *in_data, float *out_data)
{
	int i;
    
    __m256 v1, v2, vres;
     
    v1 = _mm256_load_ps(in_data);
	
	float temp[8] __attribute__((aligned(32)));
	float out_data2[8] __attribute__((aligned(32)));

	for (i = 0; i < 8; i += 4)
	{
		
        v2 = _mm256_load_ps(&(dctlookup_trans[i*8]));
        vres = _mm256_dp_ps(v1, v2, 0b11110001);
        _mm256_store_ps((float*)&temp, vres);
        out_data[i] = temp[0] + temp[4];
        
        v2 = _mm256_load_ps(&(dctlookup_trans[(i+1)*8]));
        vres = _mm256_dp_ps(v1, v2, 0b11110001);
        _mm256_store_ps((float*)&temp, vres);
        out_data[i+1] = temp[0] + temp[4];
        
        v2 = _mm256_load_ps(&(dctlookup_trans[(i+2)*8]));
        vres = _mm256_dp_ps(v1, v2, 0b11110001);
        _mm256_store_ps((float*)&temp, vres);
        out_data[i+2] = temp[0] + temp[4];
        
        v2 = _mm256_load_ps(&(dctlookup_trans[(i+3)*8]));
        vres = _mm256_dp_ps(v1, v2, 0b11110001);
        _mm256_store_ps((float*)&temp, vres);
        out_data[i+3] = temp[0] + temp[4];
    }
}

static void idct_1d(float *in_data, float *out_data)
{
  int i;
  
  __m256 v1, v2, vres;
     
    v1 = _mm256_load_ps(in_data);
	
	float temp[8] __attribute__((aligned(32)));
	float out_data2[8] __attribute__((aligned(32)));
	
    
	for (i = 0; i < 8; i += 4)
	{
		
        v2 = _mm256_load_ps(&(dctlookup[i]));
        vres = _mm256_dp_ps(v1, v2, 0b11110001);
        _mm256_store_ps((float*)&temp, vres);
        out_data[i] = temp[0] + temp[4];
        
        v2 = _mm256_load_ps(&(dctlookup[i+1]));
        vres = _mm256_dp_ps(v1, v2, 0b11110001);
        _mm256_store_ps((float*)&temp, vres);
        out_data[i+1] = temp[0] + temp[4];
        
        v2 = _mm256_load_ps(&(dctlookup[i+2]));
        vres = _mm256_dp_ps(v1, v2, 0b11110001);
        _mm256_store_ps((float*)&temp, vres);
        out_data[i+2] = temp[0] + temp[4];
        
        v2 = _mm256_load_ps(&(dctlookup[i+3]));
        vres = _mm256_dp_ps(v1, v2, 0b11110001);
        _mm256_store_ps((float*)&temp, vres);
        out_data[i+3] = temp[0] + temp[4];
  
	}
}

static void scale_block(float *in_data, float *out_data)
{
	int u, v;
	
	__m256 v_in, v_res, v_temp;
	__m256 av = _mm256_load_ps(&test1);
	__m256 au = _mm256_load_ps(&test2);
	v_temp = av;
	for (v = 0; v < 8; v += 4)
	{
		v_in = _mm256_load_ps(&in_data[v*8]);
		v_res = _mm256_mul_ps(v_in, v_temp);
		_mm256_store_ps(&out_data[v*8], v_res);
		
		v_temp = au;
		
		v_in = _mm256_load_ps(&in_data[(v+1)*8]);
		v_res = _mm256_mul_ps(v_in, v_temp);
		_mm256_store_ps(&out_data[(v+1)*8], v_res);
		
		
		v_in = _mm256_load_ps(&in_data[(v+2)*8]);
		v_res = _mm256_mul_ps(v_in, v_temp);
		_mm256_store_ps(&out_data[(v+2)*8], v_res);
		
		
		v_in = _mm256_load_ps(&in_data[(v+3)*8]);
		v_res = _mm256_mul_ps(v_in, v_temp);
		_mm256_store_ps(&out_data[(v+3)*8], v_res);
	}	
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
	int zigzag;

	__m128i vu, vv, vc;
	
	uint8_t a_v[8];
	uint8_t a_u[8];
	float adct[8] __attribute__((aligned(32)));
	float quant_table[8] __attribute__((aligned(32)));
	float out_data2[64] __attribute__((aligned(32)));
	/*
	for (zigzag = 0; zigzag < 64; zigzag += 8)
	{		
		
		int i;
		for(i = 0; i < 8; ++i)
		{
			adct[i] = in_data[(zigzag_V[zigzag+i]*8) + zigzag_U[zigzag+i]];
			quant_table[i] = 1.0f/(float)quant_tbl[zigzag+i];
		}
		
		__m256 v_dct = _mm256_load_ps(&adct);
		__m256 q_tbl = _mm256_load_ps(&quant_table);
		__m256 temp = _mm256_set1_ps(0.25f);
		temp = _mm256_mul_ps(v_dct, temp);
		temp = _mm256_round_ps(_mm256_mul_ps(temp, q_tbl), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
		_mm256_store_ps(&out_data2[zigzag], temp);
	
		/*
				(_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
				(_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
				(_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
				(_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
				_MM_FROUND_CUR_DIRECTION // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
		*
	}
	*/

	for (zigzag = 0; zigzag < 64; ++zigzag)
	{
		uint8_t u = zigzag_U[zigzag];
		uint8_t v = zigzag_V[zigzag];

		float dct = in_data[v*8+u];

		// Zig-zag and quantize //
		out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]); //_mm_cvtepi8_epi16/32
	}
	/*
	for (zigzag = 0; zigzag < 64; ++zigzag) 
	{
		if(out_data[zigzag] != out_data2[zigzag])
		{
			printf("%s: %f != %f\n", "ERROR", out_data[zigzag], out_data2[zigzag]);
		}

	}*/
}
			
static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
	int zigzag;
	
	for (zigzag = 0; zigzag < 64; ++zigzag)
	{
		uint8_t u = zigzag_U[zigzag];
		uint8_t v = zigzag_V[zigzag];

		float dct = in_data[zigzag];

		/* Zig-zag and de-quantize */
		out_data[v*8+u] = (float) round((dct * quant_tbl[zigzag]) / 4.0);
	}
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(32)));
  float mb2[8*8] __attribute((aligned(32)));
  
  int i, v;

  for (i = 0; i < 64; ++i) 
    {
        mb2[i] = in_data[i];
    }

  /* Two 1D DCT operations with transpose */
  for (v = 0; v < 8; ++v) 
    {
       dct_1d(mb2+v*8, mb+v*8);
    }
    
  transpose_block(mb, mb2);
  
  for (v = 0; v < 8; ++v)
    {
       dct_1d(mb2+v*8, mb+v*8);
    }
  transpose_block(mb, mb2);

  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (i = 0; i < 64; ++i) 
    {
        out_data[i] = mb2[i]; 
    }
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(32)));
  float mb2[8*8] __attribute((aligned(32)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb[i] = in_data[i]; }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);

  for (i = 0; i < 64; ++i) 
	{ 
		out_data[i] = mb[i];
	}
}

void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
    int v, v_stride;
       
    __m128i va, vb, v_res;
    v_res = _mm_setzero_si128();
    

    for (v = 0; v < 8; v+=4)
    {
        v_stride = v*stride;             
        //16 0 0 0 | 16 0 0 0
        
        va = _mm_loadu_si128((void const*)&block2[v_stride]);
        vb = _mm_loadu_si128((void const*)&block1[v_stride]); 
        v_res = _mm_add_epi32(v_res, _mm_sad_epu8(va, vb));
        
        v_stride += stride;
 
        va = _mm_loadu_si128((void const*)&block2[v_stride]); 
        vb = _mm_loadu_si128((void const*)&block1[v_stride]);
        v_res = _mm_add_epi32(v_res, _mm_sad_epu8(va, vb));     
        
        v_stride += stride;
 
        va = _mm_loadu_si128((void const*)&block2[v_stride]); 
        vb = _mm_loadu_si128((void const*)&block1[v_stride]);
        v_res = _mm_add_epi32(v_res, _mm_sad_epu8(va, vb));     
        
        v_stride += stride;
 
        va = _mm_loadu_si128((void const*)&block2[v_stride]); 
        vb = _mm_loadu_si128((void const*)&block1[v_stride]);
        v_res = _mm_add_epi32(v_res, _mm_sad_epu8(va, vb));          
    }
    *result = _mm_cvtsi128_si32(v_res);
  }
