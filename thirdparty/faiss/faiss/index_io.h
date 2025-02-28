/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

// I/O code for indexes

#ifndef FAISS_INDEX_IO_H
#define FAISS_INDEX_IO_H

#include <cstdio>
#include <string>
#include <typeinfo>
#include <vector>

#include <faiss/IndexIVF.h>

/** I/O functions can read/write to a filename, a file handle or to an
 * object that abstracts the medium.
 *
 * The read functions return objects that should be deallocated with
 * delete. All references within these objectes are owned by the
 * object.
 */

namespace faiss {

struct Index;
struct IndexBinary;
struct VectorTransform;
struct ProductQuantizer;
struct IOReader;
struct IOWriter;
struct InvertedLists;

void write_index(const Index* idx, const char* fname);
void write_index(const Index* idx, FILE* f);
void write_index(const Index* idx, IOWriter* writer);

void write_index_binary(const IndexBinary* idx, const char* fname);
void write_index_binary(const IndexBinary* idx, FILE* f);
void write_index_binary(const IndexBinary* idx, IOWriter* writer);

// The read_index flags are implemented only for a subset of index types.
const int IO_FLAG_READ_ONLY = 2;
// strip directory component from ondisk filename, and assume it's in
// the same directory as the index file
const int IO_FLAG_ONDISK_SAME_DIR = 4;
// don't load IVF data to RAM, only list sizes
const int IO_FLAG_SKIP_IVF_DATA = 8;
// load index data with vectors' norms
const int IO_FLAG_WITH_NORM = 1 << 8;
// try to memmap data (useful to load an ArrayInvertedLists as an
// OnDiskInvertedLists)
const int IO_FLAG_MMAP = IO_FLAG_SKIP_IVF_DATA | 0x646f0000;

Index* read_index(const char* fname, int io_flags = 0);
Index* read_index(FILE* f, int io_flags = 0);
Index* read_index(IOReader* reader, int io_flags = 0);

IndexBinary* read_index_binary(const char* fname, int io_flags = 0);
IndexBinary* read_index_binary(FILE* f, int io_flags = 0);
IndexBinary* read_index_binary(IOReader* reader, int io_flags = 0);

void write_VectorTransform(const VectorTransform* vt, const char* fname);
VectorTransform* read_VectorTransform(const char* fname);

ProductQuantizer* read_ProductQuantizer(const char* fname);
ProductQuantizer* read_ProductQuantizer(IOReader* reader);

void write_ProductQuantizer(const ProductQuantizer* pq, const char* fname);
void write_ProductQuantizer(const ProductQuantizer* pq, IOWriter* f);

void write_InvertedLists(const InvertedLists* ils, IOWriter* f);
InvertedLists* read_InvertedLists(IOReader* reader, int io_flags = 0);

// for backward compatibility
Index *read_index_nm(IOReader *f, int io_flags = 0);
void write_index_nm(const Index* idx, IOWriter* writer);
} // namespace faiss

#endif
