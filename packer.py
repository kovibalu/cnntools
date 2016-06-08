import msgpack
import msgpack_numpy
import cPickle as pickle


def packb(x, version, use_msgpack=True):
    """ Pack an object x (that can contain numpy objects) """
    if use_msgpack:
        return msgpack.packb({'version': version, 'data': x}, default=msgpack_numpy.encode)
    else:
        return pickle.dumps({'version': version, 'data': x})


def fpackb(x, version, filepath, use_msgpack=True):
    """ Pack an object x and save it to file (that can contain numpy objects) """
    packed = packb(x, version, use_msgpack)

    with open(filepath, 'w') as f:
        f.write(packed)


def unpackb(packed, use_msgpack=True):
    """ Unpack an object x (that can contain numpy objects) """
    if use_msgpack:
        dic = msgpack.unpackb(packed, object_hook=msgpack_numpy.decode)
    else:
        dic = pickle.load(packed)

    return dic['version'], dic['data']


def unpackb_version(packed, expected_version, use_msgpack=True):
    """ Unpack an object x (that can contain numpy objects) """
    if use_msgpack:
        dic = msgpack.unpackb(packed, object_hook=msgpack_numpy.decode)
    else:
        dic = pickle.load(packed)

    package_version = dic['version']

    if package_version != expected_version:
        raise ValueError('Unexpected version ({0}) when reading package (expected: {1})'.format(package_version, expected_version))

    return dic['data']


def funpackb_version(expected_version, filepath, use_msgpack=True):
    """ Unpack an object x from file (that can contain numpy objects) """
    with open(filepath, 'r') as f:
        packed = f.read()

    return unpackb_version(packed, expected_version, use_msgpack)
