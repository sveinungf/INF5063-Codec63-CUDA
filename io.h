#ifndef C63_IO_H_
#define C63_IO_H_

#include <inttypes.h>
#include <stdio.h>

#include "c63.h"

// Declarations
int read_bytes(FILE *fp, void *data, unsigned int sz);

uint16_t get_bits(struct entropy_ctx *c, uint8_t n);

uint8_t get_byte(FILE *fp);

void flush_bits(struct entropy_ctx *c);

void put_bits(struct entropy_ctx *c, uint16_t bits, uint8_t n);

void put_byte(FILE *fp, int byte);

void put_bytes(FILE *fp, const void* data, unsigned int len);

#endif  /* C63_IO_H_ */
