import os

from progressbar import ProgressBar

from cnntools.descriptor_aggregator import DescriptorAggregator
from cnntools.descstore import DescriptorStoreHdf5
from cnntools.models import CaffeCNNSnapshot
from cnntools.redis_aggregator import patch_interrupt_signal, detach_patch_interrupt_signal
from cnntools.tasks import compute_cnn_features_gpu_task
from cnntools.common_utils import iter_batch, progress_bar_widgets


def get_snapshot_id(caffe_cnn, snapshot_id):
    if snapshot_id:
        snapshot = CaffeCNNSnapshot.objects.get(id=snapshot_id)
        if snapshot.training_run.net.id != caffe_cnn.id:
            raise ValueError(
                'Snapshot\'s corresponding network id doesn\'t match given'
                ' network id!'
            )
        return snapshot_id

    # Get last snapshot
    snapshot = CaffeCNNSnapshot.objects.\
        filter(training_run__net__id=caffe_cnn.id).\
        order_by('-id')[0]

    return snapshot.id


def get_slug(caffe_cnn, snapshot_id, slug_extra):
    snapshot_id = get_snapshot_id(caffe_cnn, snapshot_id)
    slug = 'netid-{}:snapshot-{}'.format(caffe_cnn.netid, snapshot_id)
    if slug_extra:
        slug += slug_extra

    return slug, snapshot_id


def get_redis_key_core(slug, feature_name_list):
    return 'desc:slug-%s:fetname%s:batchid#' % (
        slug,
        '-'.join(feature_name_list),
    )


def get_redis_key(slug, feature_name_list, batchid_counter):
    return '%s%s' % (
        get_redis_key_core(slug, feature_name_list),
        batchid_counter,
    )


def get_task_id(slug, feature_name_list):
    return '%stask' % (
        get_redis_key_core(slug, feature_name_list),
    )


def get_descstore_dirname(netid, feature_name):
    dirname = '{}-{}'.format(netid, feature_name)
    # clean up
    invalid_chars = ['/', '(', ')', '[', ']', '{', '}', ' ', ':']

    for c in invalid_chars:
        dirname = dirname.replace(c, '-')

    return dirname


def get_descstore_filename(netid, feature_name):
    dirname = get_descstore_dirname(netid, feature_name)
    return '{}.hdf5'.format(dirname)


def get_descstore_filepaths(desc_rootpath, feature_name_list, caffe_cnn,
                            snapshot_id, slug_extra=''):
    slug, snapshot_id = get_slug(caffe_cnn, snapshot_id, slug_extra)
    return [
        os.path.join(desc_rootpath, get_descstore_filename(slug, feature_name))
        for feature_name in feature_name_list
    ]


def dispatch_feature_comp(
    desc_rootpath,
    item_type,
    item_ids,
    node_batchsize,
    feature_name_list,
    num_dims_list,
    fetcomp_func,
    fetcomp_kwargs,
    slug,
    verbose=False,
):
    if verbose:
        print 'Dispatching feature computation for incomplete {} models'.format(item_type)

    # TODO: Use DescriptorAggregator instead, so we don't have to assemble the filename...
    completed_ids = set()
    for feature_name in feature_name_list:
        filename = get_descstore_filename(slug, feature_name)
        if os.path.exists(os.path.join(desc_rootpath, filename)):
            src_store = DescriptorStoreHdf5(
                path=os.path.join(desc_rootpath, filename),
                readonly=True,
                verbose=verbose,
            )
            if verbose:
                print 'Loaded completed descriptors for {}.'.format(filename)
            # Get already completed ids
            current_ids = set(src_store.ids[...])
            if completed_ids:
                completed_ids.intersection_update(current_ids)
            else:
                completed_ids = current_ids

    if verbose:
        print 'Getting ids for incomplete items...'
    todo_ids = item_ids.difference(completed_ids)
    if verbose:
        print '{} ids completed already'.format(len(completed_ids))
        print '{} ids to do...'.format(len(todo_ids))

    task_id = get_task_id(slug, feature_name_list)

    if verbose:
        pbar = ProgressBar(widgets=progress_bar_widgets(), maxval=len(todo_ids))
        pbar_counter = 0
        pbar.start()
    batchid_counter = 0
    batch = []

    for batch in iter_batch(todo_ids, node_batchsize):
        # Dispatch job
        batch_id = get_redis_key(
            slug,
            feature_name_list,
            batchid_counter,
        )
        fetcomp_func.delay(
            item_type,
            task_id,
            batch_id,
            batch,
            feature_name_list,
            fetcomp_kwargs,
        )

        batchid_counter += 1
        if verbose:
            pbar_counter += len(batch)
            pbar.update(pbar_counter)

    if verbose:
        pbar.finish()


def aggregate_feature_comp(
    desc_rootpath,
    item_type,
    item_ids,
    feature_name_list,
    num_dims_list,
    aggr_batchsize,
    slug,
    handle_interrupt_signal=True,
    verbose=False,
):
    if verbose:
        print 'Aggregating feature computation for {} models'.format(item_type)
    if handle_interrupt_signal:
        patch_interrupt_signal()

    dirname_list = [
        get_descstore_dirname(slug, feature_name)
        for feature_name in feature_name_list
    ]

    aggregator = DescriptorAggregator(
        feature_name_list, dirname_list, num_dims_list, verbose=verbose
    )
    aggregator.load(desc_rootpath, readonly=False)
    task_id = get_task_id(slug, feature_name_list)
    ret = aggregator.run(item_ids, task_id, aggr_batchsize=aggr_batchsize)

    if handle_interrupt_signal:
        detach_patch_interrupt_signal()

    return ret


def retrieve_features(
    desc_rootpath,
    item_ids,
    feature_name_list,
    slug,
    verbose=False,
):
    ret_item_ids = item_ids is None
    features = {}
    for feature_name in feature_name_list:
        filename = get_descstore_filename(slug, feature_name)
        src_store = DescriptorStoreHdf5(
            path=os.path.join(desc_rootpath, filename),
            readonly=True,
            verbose=verbose,
        )
        # Will be called only once
        if item_ids is None:
            item_ids = src_store.ids[...]

        features[feature_name] = src_store.block_get(
            item_ids, show_progress=verbose
        )
        del src_store

    if ret_item_ids:
        return item_ids, features
    else:
        return features


def compute_features(
    desc_rootpath,
    item_type,
    item_ids,
    node_batchsize,
    aggr_batchsize,
    feature_name_list,
    num_dims_list,
    fetcomp_func,
    fetcomp_kwargs,
    slug,
    handle_interrupt_signal=True,
    verbose=False,
):
    """
    Extracts the specified feature for the specified items using a trained CNN.

    :param desc_rootpath: The root path of the directory where the computed
    features will be stored in.

    :param item_type: The type of the model class for the item which are
    classified (e.g. FgPhoto). This class should have 'title', 'photo'
    attributes/properties. The photo attribute should have most of the Photo
    model's fields. It is advised to use an actual Photo instance here.

    :param item_ids: List (or numpy array) of ids into the :ref:`item_type`
    table. The length of this list is the same as the length of :ref:`y_true`
    list and they have the same order.

    :param node_batchsize: The number of feature computations to put in one
    task executed on a worker node.

    :param aggr_batchsize: The number of batches to wait for before forcing the
    aggregator to download and remove those batches from redis.

    :param feature_name_list: The features' names in the network which will be
    extracted.

    :param num_dims_list: Dimensions of the computed features.

    :param fetcomp_func: The function which will be executed to compute
    features.

    :param fetcomp_kwargs: The parameters to pass to the feature computer
    function.

    :param slug: The unique humanly readable name associated with the feature
    computation task.

    :param handle_interrupt_signal: If True, we patch the interrupt signal so the descriptor store is saved before exiting, when the user hits Ctrl + C.

    :param verbose: If True, print progress information to the console.
    """
    item_ids_set = set(item_ids)
    single_feature = False
    if isinstance(feature_name_list, (str, unicode)):
        if isinstance(num_dims_list, list):
            raise ValueError(
                'If "feature_name_list" is not specified as a list, '
                '"num_dims_list" has to be a single number!'
            )
        feature_name_list = [feature_name_list]
        num_dims_list = [num_dims_list]
        single_feature = True

    dispatch_feature_comp(
        desc_rootpath,
        item_type,
        item_ids_set,
        node_batchsize,
        feature_name_list,
        num_dims_list,
        fetcomp_func,
        fetcomp_kwargs,
        slug,
        verbose=verbose,
    )
    was_interrupted = not aggregate_feature_comp(
        desc_rootpath,
        item_type,
        item_ids_set,
        feature_name_list,
        num_dims_list,
        aggr_batchsize,
        slug,
        handle_interrupt_signal=handle_interrupt_signal,
        verbose=verbose,
    )
    if was_interrupted:
        return None

    # The order of the item ids is important!
    fets = retrieve_features(
        desc_rootpath,
        item_ids,
        feature_name_list,
        slug,
        verbose=verbose,
    )
    if single_feature:
        fets = fets[feature_name_list[0]]

    return fets


def compute_cnn_features(
    desc_rootpath,
    item_type,
    item_ids,
    node_batchsize,
    aggr_batchsize,
    caffe_cnn,
    feature_name_list,
    num_dims_list,
    snapshot_id,
    do_preprocessing=True,
    image_dims=None,
    mean=None,
    grayscale=False,
    auto_reshape=False,
    transfer_weights=False,
    slug_extra='',
    input_trafo_func_name=None,
    input_trafo_kwargs=None,
    fet_trafo_type_id=None,
    fet_trafo_kwargs=None,
    handle_interrupt_signal=True,
    verbose=False,
):
    """
    Extracts the specified feature for the specified items using a trained CNN.

    :param desc_rootpath: The root path of the directory where the computed
    features will be stored in.

    :param item_type: The type of the model class for the item which are
    classified (e.g. FgPhoto). This class should have 'title', 'photo'
    attributes/properties. The photo attribute should have most of the Photo
    model's fields. It is advised to use an actual Photo instance here.

    :param item_ids: List (or numpy array) of ids into the :ref:`item_type`
    table. The length of this list is the same as the length of :ref:`y_true`
    list and they have the same order.

    :param node_batchsize: The number of feature computations to put in one
    task executed on a worker node.

    :param aggr_batchsize: The number of batches to wait for before forcing the
    aggregator to download and remove those batches from redis.

    :param caffe_cnn: The CaffeCNN instance which represents the CNN which is
    used for feature computation.

    :param feature_name_list: The features' names in the network which will be
    extracted.

    :param num_dims_list: Dimensions of the computed features.

    :param snapshot_id: The ID of the snapshot to use.

    :param do_preprocessing: True, if we should do preprocessing (resizing/cropping the image for example) on the input.

    :param image_dims: Tuple with two elements which defines the image
    dimensions (width, height). All input images will be resized to this size.
    This should be the same as the one was used for training.

    :param mean: Tuple with 3 elements which defines the mean which will be
    subtracted from all images. This should be the same as the one was used for
    training.

    :param grayscale: If true we convert all images to grayscale.

    :param auto_reshape: Automatically reshapes the network to the size of the
    input image. The user should still produce the same feature dimension for
    all images after the feature transformation.

    :param transfer_weights: Automatically convert the net weights to fully
    convolutional form. The deploy file of the model has to be fully
    convolutional!

    :param slug_extra: Additional string to add to the slug.

    :param input_trafo_func_name: Input (usually an image) transformation
    primitive. This should include the module and the function name like
    'parent_module.child_module.example_function_name'. This module should be
    on the python path.

    :param input_trafo_kwargs: Keyword arguments for the chosen image
    transformation primitive.

    :param fet_trafo_type_id: Feature transformation primitive. Currently the
    user can select from:
        ['MINC-spatial-avg', 'MINC-gram', 'MINC-mean-std']

    :param fet_trafo_kwargs: Keyword arguments for the chosen feature
    transformation primitive.

    :param handle_interrupt_signal: If True, we patch the interrupt signal so the descriptor store is saved before exiting, when the user hits Ctrl + C.

    :param verbose: If True, print progress information to the console.
    """
    slug, snapshot_id = get_slug(caffe_cnn, snapshot_id, slug_extra)

    fetcomp_kwargs = {
        'snapshot_id': snapshot_id,
        'do_preprocessing': do_preprocessing,
        'image_dims': image_dims,
        'mean': mean,
        'grayscale': grayscale,
        'auto_reshape': auto_reshape,
        'transfer_weights': transfer_weights,
        'input_trafo_func_name': input_trafo_func_name,
        'input_trafo_kwargs': input_trafo_kwargs,
        'fet_trafo_type_id': fet_trafo_type_id,
        'fet_trafo_kwargs': fet_trafo_kwargs,
    }

    return compute_features(
        desc_rootpath,
        item_type,
        item_ids,
        node_batchsize,
        aggr_batchsize,
        feature_name_list,
        num_dims_list,
        compute_cnn_features_gpu_task,
        fetcomp_kwargs,
        slug,
        handle_interrupt_signal=handle_interrupt_signal,
        verbose=verbose,
    )
