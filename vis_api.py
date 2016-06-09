import numpy as np

from cnntools.fetcomp import compute_cnn_features, get_snapshot_id
from resultvis.api import (process_classification_experiment,
                           process_regression_experiment,
                           process_tagging_experiment)


def generate_classification_results(desc_rootpath, label_type, label_ids,
                                    item_type, item_ids, y_true,
                                    feature_name, node_batchsize,
                                    aggr_batchsize, caffe_cnn, snapshot_id,
                                    image_dims, mean, class_setup,
                                    description="", url_slug=None):
    """
    Generates results for a trained CNN and submits it to into resultvis system

    :param desc_rootpath: The root path of the directory where the computed
    features will be stored in.

    :param label_type: The type of the model class for the labels which are
    used for classification (e.g. SubstanceCategory). This class should have
    the __str__ method implemented.

    :param label_ids: List (or numpy array) of ids into the :ref:`label_type`
    table. The length of this list is the number of different classes/labels.
    The index of the id determines the labels in the :ref:`y_true` parameter.
    (e.g. the list is [21, 34], and the names of SubstanceCategories with
    these ids are 'fabric' and 'metal', then label 0 in y_true will mean
    'fabric' and label 1 will mean 'metal' similarly).

    :param item_type: The type of the model class for the item which are
    classified (e.g. FgPhoto). This class should have 'title', 'photo'
    attributes/properties. The photo attribute should have most of the Photo
    model's fields. It is advised to use an actual Photo instance here.

    :param item_ids: List (or numpy array) of ids into the :ref:`item_type`
    table. The length of this list is the same as the length of :ref:`y_true`
    list and they have the same order.

    :param y_true: List (or numpy array) of true labels for the test set.
    Each element is an integer as explained above at :ref:`label_type`.

    :param feature_name: Output name in the network which represents
    the predicted class probabilities.

    :param node_batchsize: The number of feature computations to put in one
    task executed on a worker node.

    :param aggr_batchsize: The number of batches to wait for before forcing
    the aggregator to download and remove those batches from redis.

    :param caffe_cnn: The CaffeCNN instance which represents the CNN which is
    used for feature computation.

    :param snapshot_id: The ID of the snapshot to use. If None, we will load
    the latest snapshot.

    :param image_dims: Tuple with two elements which defines the image
    dimensions (width, height). All input images will be resized to this
    size.
    This should be the same as the one was used for training.

    :param mean: Tuple with 3 elements which defines the mean which will be
    subtracted from all images. This should be the same as the one was used
    for training.

    :param class_setup: ClassificationSetup instance to save the results
    under.

    :param description: Description for the ClassificationResult to be
    created.

    :param url_slug: For modularity, we store the url which can be used to
    reach detailed information about classified items.
    """
    num_dims = len(label_ids)

    snapshot_id = get_snapshot_id(caffe_cnn, snapshot_id)
    fets = compute_cnn_features(
        desc_rootpath,
        item_type,
        item_ids,
        node_batchsize,
        aggr_batchsize,
        caffe_cnn,
        [feature_name],
        [num_dims],
        snapshot_id,
        image_dims,
        mean,
    )
    # The aggregation was terminated
    if fets is None:
        return

    probs = fets[feature_name]

    # Choose the highest probability as the predicted label
    y_pred = np.argmax(probs, axis=1)

    process_classification_experiment(
        label_type=label_type,
        label_ids=label_ids,
        item_type=item_type,
        item_ids=item_ids,
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        class_setup=class_setup,
        description=description,
        url_slug=url_slug,
        snapshot_id=snapshot_id,
    )


def generate_tagging_results(desc_rootpath, label_type, label_ids, item_type,
                             item_ids, y_true, feature_name, node_batchsize,
                             aggr_batchsize, caffe_cnn, snapshot_id,
                             image_dims, mean, class_setup,
                             description="", url_slug=None):
    """
    Generates results for a trained CNN and submits it to into resultvis system

    :param desc_rootpath: The root path of the directory where the computed
    features will be stored in.

    :param label_type: The type of the model class for the labels which are
    used for classification (e.g. SubstanceCategory). This class should have
    the __str__ method implemented.

    :param label_ids: List (or numpy array) of ids into the :ref:`label_type`
    table. The length of this list is the number of different classes/labels.
    The index of the id determines the labels in the :ref:`y_true` parameter.
    (e.g. the list is [21, 34], and the names of SubstanceCategories with these
    ids are 'fabric' and 'metal', then label 0 in y_true will mean 'fabric' and
    label 1 will mean 'metal' similarly).

    :param item_type: The type of the model class for the item which are
    classified (e.g. FgPhoto). This class should have 'title', 'photo'
    attributes/properties. The photo attribute should have most of the Photo
    model's fields. It is advised to use an actual Photo instance here.

    :param item_ids: List (or numpy array) of ids into the :ref:`item_type`
    table. The length of this list is the same as the length of :ref:`y_true`
    list and they have the same order.

    :param y_true: 2D numpy array of true tags for the test set in binary
    indicator form. Each row corresponds to an item (the order is the same as
    item_ids) and each column corresponds to a label (tag) type. If the value
    is 1 then the current item is labeled with the current label. All other
    values should be 0.

    :param feature_name: Output name in the network which represents
    the predicted class probabilities.

    :param node_batchsize: The number of feature computations to put in one
    task executed on a worker node.

    :param aggr_batchsize: The number of batches to wait for before forcing the
    aggregator to download and remove those batches from redis.

    :param caffe_cnn: The CaffeCNN instance which represents the CNN which is
    used for feature computation.

    :param snapshot_id: The ID of the snapshot to use. If None, we will load
    the latest snapshot.

    :param image_dims: Tuple with two elements which defines the image
    dimensions (width, height). All input images will be resized to this size.
    This should be the same as the one was used for training.

    :param mean: Tuple with 3 elements which defines the mean which will be
    subtracted from all images. This should be the same as the one was used for
    training.

    :param class_setup: ClassificationSetup instance to save the results under.

    :param description: Description for the ClassificationResult to be created.

    :param url_slug: For modularity, we store the url which can be used to
    reach detailed information about classified items.
    """
    num_dims = len(label_ids)

    snapshot_id = get_snapshot_id(caffe_cnn, snapshot_id)
    fets = compute_cnn_features(
        desc_rootpath,
        item_type,
        item_ids,
        node_batchsize,
        aggr_batchsize,
        caffe_cnn,
        [feature_name],
        [num_dims],
        snapshot_id,
        image_dims,
        mean,
    )
    # The aggregation was terminated
    if fets is None:
        return

    probs = fets[feature_name]

    process_tagging_experiment(
        label_type,
        label_ids,
        item_type,
        item_ids,
        y_true,
        probs,
        class_setup,
        description,
        url_slug,
        snapshot_id,
    )


def generate_regression_results(desc_rootpath, item_type, item_ids, y_true,
                                feature_name, num_dims, node_batchsize,
                                aggr_batchsize, caffe_cnn, snapshot_id,
                                image_dims, mean, class_setup,
                                description="", url_slug=None, fet_trafo_func=None):
    """
    Generates results for a trained CNN and submits it to into resultvis system

    :param desc_rootpath: The root path of the directory where the computed
    features will be stored in.

    :param item_type: The type of the model class for the item which are
    classified (e.g. FgPhoto). This class should have 'title', 'photo'
    attributes/properties. The photo attribute should have most of the Photo
    model's fields. It is advised to use an actual Photo instance here.

    :param item_ids: List (or numpy array) of ids into the :ref:`item_type`
    table. The length of this list is the same as the length of :ref:`y_true`
    list and they have the same order.

    :param y_true: List or numpy array of true regression values for the test
    set. Each row corresponds to an item (the order is the same as item_ids).

    :param feature_name: Output name in the network which represents
    the predicted class probabilities.

    :param num_dims: Dimension of the computed feature.

    :param node_batchsize: The number of feature computations to put in one
    task executed on a worker node.

    :param aggr_batchsize: The number of batches to wait for before forcing the
    aggregator to download and remove those batches from redis.

    :param caffe_cnn: The CaffeCNN instance which represents the CNN which is
    used for feature computation.

    :param snapshot_id: The ID of the snapshot to use. If None, we will load
    the latest snapshot.

    :param image_dims: Tuple with two elements which defines the image
    dimensions (width, height). All input images will be resized to this size.
    This should be the same as the one was used for training.

    :param mean: Tuple with 3 elements which defines the mean which will be
    subtracted from all images. This should be the same as the one was used for
    training.

    :param class_setup: ClassificationSetup instance to save the results under.

    :param description: Description for the ClassificationResult to be created.

    :param url_slug: For modularity, we store the url which can be used to
    reach detailed information about classified items.

    :param fet_trafo_func: Transform the features after extracting them. By
    default this function is None, so we don't do any transformation.
    """
    snapshot_id = get_snapshot_id(caffe_cnn, snapshot_id)
    fets = compute_cnn_features(
        desc_rootpath,
        item_type,
        item_ids,
        node_batchsize,
        aggr_batchsize,
        caffe_cnn,
        [feature_name],
        [num_dims],
        snapshot_id,
        image_dims,
        mean,
    )
    # The aggregation was terminated
    if fets is None:
        return

    # Transform features if we have transformation defined
    if fet_trafo_func:
        fets = fet_trafo_func(fets[feature_name])

    process_regression_experiment(
        item_type,
        item_ids,
        y_true,
        fets,
        class_setup,
        description,
        url_slug,
        snapshot_id,
    )
