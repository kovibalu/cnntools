# The MIT License (MIT)
#
# Copyright (c) 2016 Sean Bell
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Based on https://github.com/seanbell/descriptor-store
#
# The MIT License (MIT)
#
# Copyright (c) 2016 Balazs Kovacs (modified the original implementation)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
import random
import shutil
import tempfile
import time

import numpy as np

import h5py
from cnntools.common_utils import progress_bar


def hdf5_to_memmap(src_path, dst_path, dst_data_dtype=None, verbose=True):
    assert os.path.exists(src_path)
    assert not os.path.exists(dst_path) or os.path.isdir(dst_path)

    if verbose:
        print "load src..."
    src = DescriptorStoreHdf5(path=src_path, readonly=True)
    if verbose:
        print "create dst..."
    dst = DescriptorStoreMemmap(path=dst_path, readonly=False)
    if not dst_data_dtype:
        dst_data_dtype = src.data.dtype
    dst.create(max_ids=src.num_ids,
               num_dims=src.num_dims,
               id_dtype=src.ids.dtype,
               data_dtype=dst_data_dtype)
    if verbose:
        print "copy IDs..."
    dst._ids[:] = sorted(src.ids[:])
    dst._next_idx = src.num_ids
    if verbose:
        print "copy data..."
    src.block_get(dst.ids, ret=dst._data, show_progress=verbose)
    if verbose:
        print "reconstruct"
    dst.reconstruct()

    if verbose:
        print "check random IDs..."
    del src
    del dst
    src = DescriptorStoreHdf5(path=src_path, readonly=True)
    dst = DescriptorStoreMemmap(path=dst_path, readonly=True)
    ids = random.sample(src.ids[:], min(1200, src.num_ids))
    if np.linalg.norm(src.block_get(ids) - dst.block_get(ids)) >= 1e-6 * len(ids):
        print "Error!  Descriptor mismatch"


class DescriptorStoreHdf5(object):

    def __init__(self, path, readonly=True, verbose=True):
        self._path = path
        self._readonly = readonly
        self.verbose = verbose
        exists = os.path.exists(self._path)
        if self.verbose:
            print "DescriptorStoreHdf5.__init__(path='%s', readonly=%s), exists: %s" % (
                path, readonly, exists)

        self._created = False
        if readonly:
            if exists:
                self._file = h5py.File(self._path, mode='r')
                self._load()
            else:
                raise IOError("File does not exist: '%s'" % path)
        else:
            if exists:
                self._file = h5py.File(self._path, mode='r+')
                self._load()
            else:
                self._file = h5py.File(self._path, mode='w')

        if readonly:
            assert self._created, "Could not load '%s'" % path

    def _load(self):
        if self.verbose:
            print "DescriptorStoreHdf5._load... "
        start_time = time.time()
        self._ids = self._file['ids']
        self._data = self._file['data']
        self._update_map()
        self._created = True
        if self.verbose:
            print "DescriptorStoreHdf5._load: ids: %s, data: %s (%.3f s)" % (
                self._ids.shape, self._data.shape, time.time() - start_time)

    def create(self, num_dims, id_dtype=np.int64, data_dtype=np.float32):
        if self.verbose:
            print "DescriptorStoreHdf5.create(num_dims=%s, id_type=%s, data_type=%s)..." % (
                num_dims, id_dtype, data_dtype)

        if 'ids' in self._file:
            if self.verbose:
                print "DescriptorStoreHdf5.create: deleting existing ids"
            del self._file['ids']
        if 'data' in self._file:
            if self.verbose:
                print "DescriptorStoreHdf5.create: deleting existing data"
            del self._file['data']

        opts = dict(
            shuffle=True,
            fletcher32=True,
            compression="lzf",
        )
        self._ids = self._file.create_dataset(
            name='ids', dtype=id_dtype, shape=(0, ),
            maxshape=(None, ), chunks=(16384, ),
            fillvalue=0, **opts)

        row_chunks = np.clip(int(16 * 4096 / num_dims), 1, 16384)
        self._data = self._file.create_dataset(
            name='data', dtype=data_dtype, shape=(0, num_dims),
            maxshape=(None, num_dims), chunks=(row_chunks, num_dims),
            fillvalue=float('nan'), **opts)

        self._update_map()

    def save_dataset(self, name, data):
        d = None
        if name in self._file:
            d = self._file[name]
            if np.dtype(d.dtype) != np.dtype(data.dtype) or d.shape != data.shape:
                del self._file[name]
                d = None
        if d is None:
            d = self._file.create_dataset(name=name, dtype=data.dtype, shape=data.shape)
        d[...] = data

    def get_dataset(self, name):
        if name in self._file:
            return self._file[name]
        else:
            return None

    def block_append(self, ids, data):
        """ Efficiently append a block of ids and data.  All IDs must be new. """
        assert ids.ndim == 1 and data.ndim == 2
        assert data.shape == (ids.shape[0], self._data.shape[1])
        assert not any(self.has_id(id) for id in ids)

        old_rows = self._data.shape[0]
        new_rows = old_rows + data.shape[0]
        self._data.resize(new_rows, axis=0)
        self._data[old_rows:new_rows, :] = data
        self._ids.resize(new_rows, axis=0)
        self._ids[old_rows:new_rows] = ids
        for i in xrange(ids.shape[0]):
            self._id_to_idx[ids[i]] = old_rows + i

        # debug
        #tmp = self._id_to_idx.copy()
        #self._update_map()
        #assert tmp == self._id_to_idx

    def set(self, id, value):
        """ This method is inefficient -- use a DescriptorStoreHdf5Buffer or
        use block_append """
        if id in self._id_to_idx:
            # see https://github.com/h5py/h5py/issues/492
            idx = self._id_to_idx[id]
            self._data[idx:idx+1, :] = value.ravel()
        else:
            self.block_append(np.asarray([id]), value.reshape(1, self._data.shape[1]))

    def get(self, id):
        """ Returns a single vector by id; this method is inefficient -- use block_get. """
        idx = self._id_to_idx[id]
        return self._data[idx, :]

    def block_get(self, ids, dtype=None, ret=None, batchsize=512, show_progress=False):
        """ Efficiently fetch a block of descriptors by IDs, optionally
        converting them to another dtype. """
        if ret is None:
            ret_dtype = dtype if dtype else self._data.dtype
            ret = np.empty((len(ids), self._data.shape[1]), dtype=ret_dtype)
        if len(ids):
            # Sorting and batching is necessary due to the requirments of "fancy indexing"
            # (http://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing).
            indices = np.array([self._id_to_idx[id] for id in ids])
            order = np.argsort(indices)
            for i in progress_bar(xrange(0, len(ids), batchsize), show_progress=show_progress):
                sub_order = order[i:i+batchsize]
                ret[sub_order, :] = self._data[indices[sub_order], :]
        return ret

    def has_id(self, id):
        return (id in self._id_to_idx)

    @property
    def path(self):
        return self._path

    @property
    def created(self):
        return self._created

    @property
    def ids(self):
        return self._ids

    @property
    def data(self):
        return self._data

    @property
    def num_dims(self):
        return self._data.shape[1]

    @property
    def num_ids(self):
        return self._ids.shape[0]

    def flush(self):
        if not self._readonly:
            if self.verbose:
                print "DescriptorStoreHdf5.flush: %s..." % self._path
            start_time = time.time()
            self._file.flush()
            if self.verbose:
                print "DescriptorStoreHdf5.flush: %s done (%.3f s)" % (
                    self._path, time.time() - start_time)

    def __del__(self):
        if self._created:
            self.flush()
            self._file.close()

    def _update_map(self):
        self._id_to_idx = {id: idx for idx, id in enumerate(self._ids[...])}
        assert len(self._id_to_idx) == self._ids.shape[0]
        assert len(self._id_to_idx) == self._data.shape[0]


class DescriptorStoreHdf5Buffer(object):
    """ In-memory buffer that batches writes for block updates to HDF5.  NOTE:
    For efficiency, it drops duplicate writes to existing ids -- the second
    write is ignored. """

    def __init__(self, store, buffer_size=65536, verbose=True):
        self._store = store
        self._buffer_size = buffer_size
        self._pending_ids = np.empty((buffer_size, ), dtype=store.ids.dtype)
        self._pending_data = np.empty((buffer_size, store.num_dims), dtype=store.data.dtype)
        self.verbose = verbose
        #self._pending_ids.fill(0)
        #self._pending_data.fill(np.nan)
        self._size = 0
        self._ids_set = set(store.ids[...])

    def set(self, id, value, force=False):
        '''If force is True, we directly write to the HDF5 store. Beware, this
        is very slow!'''
        assert id > 0
        if force:
            self._store.set(id, value)
            return

        if id not in self._ids_set:
            assert self._size < self._pending_ids.shape[0]
            self._pending_ids[self._size] = id
            self._pending_data[self._size, :] = value.ravel()
            self._size += 1
            self._ids_set.add(id)
            if self._size >= self._pending_ids.shape[0]:
                self.flush()

    def get(self, id):
        """ Warning: this is inefficient """
        for i, p_id in enumerate(self._pending_ids):
            if id == p_id:
                return self._pending_data[i, :]
        return self._store.get(id)

    def flush(self):
        if self._size == 0:
            return

        start_time = time.time()
        ordering = np.argsort(self._pending_ids[:self._size])
        self._store.block_append(
            self._pending_ids[ordering],
            self._pending_data[ordering, :])

        num_appended = self._size
        #self._pending_ids.fill(0)
        #self._pending_data.fill(np.nan)
        self._size = 0

        elapsed_time = time.time() - start_time
        if elapsed_time > 10 and self.verbose:
            print "DescriptorStoreHdf5Buffer.flush: block-append %s items (%.3f s, logging because > 10s)" % (
                num_appended, elapsed_time)

    def has_id(self, id):
        return (id in self._ids_set)

    def __del__(self):
        self.flush()


class DescriptorStoreMemmap(object):
    """ Store a matrix of feature descriptors of shape (max_ids, num_dims).
    """

    _META_ATTRS = ('_max_ids', '_num_dims', '_data_dtype', '_id_dtype', '_created', '_next_idx')

    def __init__(self, path, readonly=False):
        self._path = path
        self._readonly = readonly
        self._meta_filename = os.path.join(self._path, 'meta.json')
        self._ids_filename = os.path.join(self._path, 'ids.npy')
        self._data_filename = os.path.join(self._path, 'data.npy')
        self._created = False
        self._dirty = False

        if os.path.exists(self._meta_filename):
            self._load_meta()
            self._load_data()

        if readonly:
            assert self._created and not self._dirty, "Could not open %s" % path

    def create(self, max_ids, num_dims,
               id_dtype='int64', data_dtype='float32'):
        if os.path.exists(self._path):
            assert os.path.isdir(self._path)
        else:
            os.makedirs(self._path)

        self._max_ids = max_ids
        self._num_dims = num_dims
        self._data_dtype = _dtype_to_str(data_dtype)
        self._id_dtype = _dtype_to_str(id_dtype)

        self._load_data(initialize=True)
        self._created = True
        self._save_meta()

    def save_dataset(self, name, data):
        fname = os.path.join(self._path, '%s.npy' % name)
        np.save(fname, data)

    def get_dataset(self, name, mmap_mode=None):
        fname = os.path.join(self._path, '%s.npy' % name)
        if os.path.exists(fname):
            return np.load(fname, mmap_mode=mmap_mode)
        else:
            return None

    def has_id(self, id):
        return id in self._id_to_idx

    @property
    def ids(self):
        if self._next_idx < self._ids.shape[0]:
            return self._ids[:self._next_idx]
        else:
            return self._ids

    @property
    def data(self):
        if self._next_idx < self._data.shape[0]:
            return self._data[:self._next_idx, :]
        else:
            return self._data

    def get(self, id):
        return self._data[self._id_to_idx[id], :]

    def block_get(self, ids, dtype=None, show_progress=False):
        indices = [
            self._id_to_idx[id]
            for id in progress_bar(ids, show_progress=show_progress)
        ]
        ret = self._data[indices, :]
        if dtype:
            ret = ret.astype(dtype)
        return ret

    def set(self, id, value):
        assert not self._readonly
        assert id
        assert value.size == self.num_dims
        assert not np.isnan(value).any()

        if id in self._id_to_idx:
            idx = self._id_to_idx[id]
        else:
            idx = self._next_idx
            self._next_idx += 1
        assert idx < self._max_ids

        self._data[idx, :] = value.ravel()
        self._ids[idx] = id
        self._id_to_idx[id] = idx
        self._dirty = True

        assert self._next_idx == len(self._id_to_idx)

    def reconstruct(self):
        assert not self._readonly
        assert self._ids.ndim == 1 and self._ids.shape[0] >= self._next_idx
        assert self._data.shape == (self._ids.shape[0], self._num_dims)
        assert np.all(self._ids[:self._next_idx] != 0)
        self._id_to_idx = {
            self._ids[idx]: idx
            for idx in xrange(self._next_idx)
        }
        self._dirty = True
        self.flush()

    def __del__(self):
        if self._created and self._dirty:
            self.flush()

    def flush(self):
        if self.verbose:
            print 'flush: %s...' % self._meta_filename
        self._data.flush()
        self._ids.flush()
        self._save_meta()
        self._dirty = False
        if self.verbose:
            print 'flush complete: %s' % self._meta_filename

    @property
    def created(self):
        return self._created

    @property
    def max_ids(self):
        return self._max_ids

    @property
    def num_ids(self):
        return self._next_idx

    @property
    def path(self):
        return self._path

    @property
    def num_dims(self):
        return self._num_dims

    def _save_meta(self):
        meta = {
            attr: getattr(self, attr)
            for attr in DescriptorStoreMemmap._META_ATTRS
        }
        json.dump(meta, open(self._meta_filename, 'w'))

    def _load_meta(self):
        print "DescriptorStoreMemmap._load_meta: loading '%s'" % self._meta_filename
        meta = json.load(open(self._meta_filename))
        print "DescriptorStoreMemmap._load_meta: meta: %s" % meta
        for attr in DescriptorStoreMemmap._META_ATTRS:
            setattr(self, attr, meta[attr])

    def _load_data(self, initialize=False):
        if not initialize:
            assert os.path.exists(self._data_filename)
            assert os.path.exists(self._ids_filename)

        if self._readonly:
            mode = 'r'
        else:
            mode = 'w+' if initialize else 'r+'

        if self.verbose:
            print "DescriptorStoreMemmap._load_data: loading '%s', shape: (%s, %s), mode: %s" % (
            self._data_filename, self._max_ids, self._num_dims, mode)
        self._data = np.memmap(
            filename=self._data_filename,
            dtype=self._data_dtype,
            mode=mode,
            shape=(self._max_ids, self._num_dims),
        )

        if self.verbose:
            print "DescriptorStoreMemmap._load_data: loading '%s', shape: (%s, ), mode: %s" % (
            self._ids_filename, self._max_ids, mode)
        self._ids = np.memmap(
            filename=self._ids_filename,
            dtype=self._id_dtype,
            mode=mode,
            shape=(self._max_ids, ),
        )

        if initialize:
            self._dirty = True
            self._next_idx = 0
            #print "Filling data with nan..."
            #self._data.fill(float('nan'))
            #self._data.flush()
            if self.verbose:
                print "Filling ids with 0..."
            self._ids.fill(0)
            self._ids.flush()

        self._id_to_idx = {}
        for idx in xrange(self._next_idx):
            if self._ids[idx]:
                self._id_to_idx[self._ids[idx]] = idx
            else:
                print "Warning: _next_idx smaller than expected (actual: %s, expected: %s)" % (
                    idx, self._next_idx)
                self._next_idx = idx
                break

        if self.verbose:
            print "DescriptorStoreMemmap._load_data: _next_idx: %s" % self._next_idx
        assert len(self._id_to_idx) == self._next_idx


def _dtype_to_str(dtype):
    if isinstance(dtype, basestring):
        return dtype
    elif isinstance(dtype, type):
        return dtype.__name__
    elif hasattr(dtype, 'name'):
        return dtype.name
    else:
        raise ValueError("Not a dtype: '%s'" % repr(dtype))


def _test_store_all():
    _test_store(DescriptorStoreHdf5)
    _test_store(DescriptorStoreMemmap)


def _test_store(StoreType):
    print "_test_store..."
    expected = {}
    num_dims = 3999
    num_ids = 6231

    root_path = tempfile.mkdtemp()
    if StoreType == DescriptorStoreHdf5:
        path = os.path.join(root_path, 'test.hdf5')
    else:
        path = root_path

    try:
        store = StoreType(path=path, readonly=False)
        if StoreType == DescriptorStoreHdf5:
            store.create(num_dims=num_dims)
            store = DescriptorStoreHdf5Buffer(store, buffer_size=5)
        else:
            store.create(max_ids=num_ids, num_dims=num_dims)

        ids = range(1, num_ids * 2)
        random.shuffle(ids)
        ids = ids[:num_ids]
        for _ in xrange(3):
            random.shuffle(ids)
            for id in ids:
                if random.randint(0, 1000) == 0:
                    print "Random reload"
                    del store
                    store = StoreType(path=path, readonly=False)
                value = np.random.randn(num_dims)
                store.set(id, value)
                expected[id] = value
                assert np.linalg.norm(expected[id] - store.get(id)) < 1e-5
            for id in ids:
                assert np.linalg.norm(expected[id] - store.get(id)) < 1e-5
            del store
            store = StoreType(path=path, readonly=False)
            for id in ids:
                assert np.linalg.norm(expected[id] - store.get(id)) < 1e-5

            if StoreType == DescriptorStoreHdf5:
                for l in (0, 1, 511, 512, 513, 1023, 1024, 1025, len(ids)):
                    print "Testing block size: %s" % l
                    test_ids = list(ids)
                    random.shuffle(test_ids)
                    test_ids = test_ids[:l]
                    block_data = store.block_get(test_ids)
                    assert block_data.shape == (len(test_ids), num_dims)
                    for i, id in enumerate(test_ids):
                        assert np.linalg.norm(expected[id] - block_data[i, :]) < 1e-5
    finally:
        shutil.rmtree(root_path)
    print "_test_store: passed."
