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

import os

import numpy as np

import cnntools
from cnntools.descstore import (DescriptorStoreHdf5, DescriptorStoreHdf5Buffer,
                                hdf5_to_memmap)
from cnntools.redis_aggregator import RedisAggregator


class DescriptorAggregator(RedisAggregator):
    def __init__(self, feature_name_list, filename_list, num_dims_list, postprocess=False):
        RedisAggregator.__init__(self)
        self.feature_name_list = feature_name_list
        self.filename_list = filename_list
        self.num_dims_list = num_dims_list
        self.postprocess = postprocess
        assert len(self.filename_list) == len(self.num_dims_list)
        assert len(self.filename_list) == len(self.feature_name_list)

    def load(self, rootpath, readonly=True):
        self.hdf5_filepath_list = []
        self.hdf5_dirpath_list = []
        self.store_list = []

        for filename, num_dims in zip(self.filename_list, self.num_dims_list):
            hdf5_filepath = os.path.join(rootpath, '{}.hdf5'.format(filename))
            hdf5_dirpath = os.path.join(rootpath, filename)

            store = DescriptorStoreHdf5(
                path=hdf5_filepath,
                readonly=readonly,
            )
            if not store.created:
                store.create(
                    num_dims=num_dims, id_dtype=np.int64, data_dtype=np.float32
                )
            assert store.num_dims == num_dims

            self.hdf5_filepath_list.append(hdf5_filepath)
            self.hdf5_dirpath_list.append(hdf5_dirpath)
            self.store_list.append(store)

    def run(self, all_ids, task_id, aggr_batchsize):
        '''Returns False if it was interrupted'''
        self.store_buffer_list = [
            DescriptorStoreHdf5Buffer(store, buffer_size=65536)
            for store in self.store_list
        ]

        # The intersection of all computed ids is fully complete
        completed_ids = set()
        for store in self.store_list:
            current_ids = set(store.ids[...])
            if completed_ids:
                completed_ids.intersection_update(current_ids)
            else:
                completed_ids = current_ids

        all_ids.update(completed_ids)
        num_ids = len(all_ids)

        print "starting scan"
        self.scan(task_id, completed_ids, num_ids, aggr_batchsize)
        print "scan exited"

        self.flush()

        if cnntools.redis_aggregator.INTERRUPTED:
            print "INTERRUPTED -- skipping hdf5_to_memmap..."
            return False

        if self.postprocess:
            print "hdf5_to_memmap..."
            for hdf5_filepath, hdf5_dirpath in zip(self.hdf5_filepath_list, self.hdf5_dirpath_list):
                hdf5_to_memmap(
                    src_path=hdf5_filepath,
                    dst_path=hdf5_dirpath,
                    dst_data_dtype='float32'
                )
        return True

    def close_store(self):
        for store in self.store_list:
            del store

    def flush(self):
        print "Flushing buffers"
        for store_buffer, store in zip(self.store_buffer_list, self.store_list):
            store_buffer.flush()
            store.flush()

    def aggregate_item(self, value):
        '''
        Input:
            value -- A list which contains the model_class ids and the computed
            features as tuples. The computed features should be stored as a
            dictionary (key: feature_name, value: feature as numpy array)
        Output:
            Return the added ids if successful, return empty list on failure
        '''
        ret_ids = set()

        for obj_id, fet_dic in value:
            if obj_id not in self.completed_ids:
                for feature_name, num_dims, store_buffer in zip(self.feature_name_list, self.num_dims_list, self.store_buffer_list):
                    fet = fet_dic[feature_name]
                    if fet.size == num_dims:
                        store_buffer.set(obj_id, fet)
                    else:
                        # Print error, but still add obj_id to the list, so we will delete it from redis...
                        print "Error: dims mismatch %s" % fet.size
            ret_ids.add(obj_id)
        return ret_ids
