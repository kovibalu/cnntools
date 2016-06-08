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

import signal
import time
import traceback

import redis
from cnntools import packer
from cnntools.common_utils import progress_bar_widgets
from django.conf import settings
from progressbar import ProgressBar

INTERRUPTED = False


def patch_interrupt_signal():
    def signal_handler(signal, frame):
        global INTERRUPTED
        print('\nInterrupted -- will exit on next loop iteration.')
        INTERRUPTED = True
    signal.signal(signal.SIGINT, signal_handler)


def detach_patch_interrupt_signal():
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def batch_ready(task_id, batch_id, value):
    client = redis.StrictRedis(**settings.REDIS_AGGRO_LOCAL_CONFIG)
    client.set(batch_id, value)
    client.sadd(task_id, batch_id)


class RedisAggregator(object):

    def scan(self, task_id, completed_ids, num_ids, aggr_batchsize=1024, flush_interval=60):
        self.task_id = task_id
        self.num_ids = num_ids
        self.aggr_batchsize = aggr_batchsize
        self.completed_ids = set(completed_ids)

        print "RedisAggregator.scan(task_id: %s, completed_ids: %s, num_ids: %s)" % (
            self.task_id, len(self.completed_ids), self.num_ids)

        self.client = redis.StrictRedis(**settings.REDIS_AGGRO_LOCAL_CONFIG)
        print "initial dbsize: %s" % self.client.dbsize()

        self.visited_keys = set()
        self.keys_to_delete = set()
        self.keys_to_get = []

        self.last_flush = time.time()

        num_remaining = self.num_ids - len(self.completed_ids)
        self.pbar = ProgressBar(
            widgets=progress_bar_widgets(),
            maxval=num_remaining,
        )
        self.pbar.start()
        self.pbar_count = 0
        while self.pbar_count < num_remaining and not INTERRUPTED:
            try:
                self.scan_for_keys()

                if time.time() > self.last_flush + flush_interval:
                    self.flush()
                    self.last_flush = time.time()
            except Exception:
                traceback.print_exc()
                time.sleep(10)

        # if we finished, delete set which holds the completed keys
        if not INTERRUPTED:
            self.client.delete(self.task_id)

        self.pbar.finish()

    def aggregate_item(self, value):
        """ Return the added ids if successful, empty list if unsuccessful """
        raise NotImplementedError()

    def flush(self):
        """ Override this if you want flush behavior """
        pass

    def scan_for_keys(self):
        """ Scan for available keys on redis """
        finished_keys = self.client.smembers(self.task_id)
        for key in finished_keys:
            if key not in self.visited_keys:
                self.visited_keys.add(key)
                self.keys_to_get.append(key)
                if len(self.keys_to_get) >= self.aggr_batchsize:
                    self.batch_get()
                if len(self.keys_to_delete) >= self.aggr_batchsize:
                    self.batch_delete()
            if INTERRUPTED:
                return
        made_progress = (self.keys_to_get or self.keys_to_delete)
        self.batch_get()
        self.batch_delete()
        if not made_progress:
            time.sleep(4)

    def unpack_task_value(self, value):
        value = packer.unpackb_version(value, settings.API_VERSION)
        #if value['api_version'] == settings.API_VERSION:
            #return value['data']
        #return None
        return value

    def batch_get(self):
        """ Process a batch of keys """
        if self.keys_to_get:
            values = self.client.mget(self.keys_to_get)
            for i, key in enumerate(self.keys_to_get):
                try:
                    data = self.unpack_task_value(values[i])
                    added_ids = self.aggregate_item(data)
                    if len(added_ids) > 0:
                        self.pbar_count += len(added_ids)
                        self.completed_ids.update(added_ids)
                        self.pbar.update(self.pbar_count)
                except Exception:
                    traceback.print_exc()

                if INTERRUPTED:
                    return
            self.keys_to_delete.update(self.keys_to_get)
            self.keys_to_get = []

    def batch_delete(self):
        """ Delete a batch of keys """
        if self.keys_to_delete:
            self.client.srem(self.task_id, *self.keys_to_delete)
            self.client.delete(*self.keys_to_delete)
            self.keys_to_delete = set()

