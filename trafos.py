import numpy as np

from scipy.ndimage.interpolation import zoom


def dummy_image_trafo_func(item, params):
    '''This is just a dummy function to show the parameters we expect in a
    image transformation function

    :param item: A database object which has an associated image
    :param params: A dictionary with parameters which might specify how we
    perform the transformation

    :return The retrieved and transformed image
    '''
    raise NotImplemented()


def fet_avg_trafo(fet, mask_resized):
    '''Feature dimensions: C x H x W, mask dimensions: H x W'''
    mask_resized = mask_resized[np.newaxis, :, :]
    num = np.sum(fet * mask_resized, axis=(1, 2))
    denom = np.sum(mask_resized, axis=(1, 2))
    fet_avg = num / denom

    return fet_avg


def comp_gram(fet, mask_resized):
    '''Feature dimensions: C x H x W, mask dimensions: H x W'''
    mask_resized = mask_resized[np.newaxis, :, :]
    # Set masked out feature activations to zero, so they don't modify the gram matrix values
    fet[~mask_resized] = 0
    fet_mx = np.reshape(
        fet,
        (fet.shape[0], fet.shape[1] * fet.shape[2])
    )
    return np.ravel(np.dot(fet_mx, fet_mx.T))


def comp_mean_std(fet, mask_resized):
    '''Feature dimensions: C x H x W, mask dimensions: H x W'''
    fet = np.reshape(fet, (fet.shape[0], -1))
    mask_resized = np.ravel(mask_resized)

    # If the mask is all zeros, just return zeros as features
    if np.sum(mask_resized) == 0:
        return np.zeros((fet.shape[0] * 2))

    fet_filt = fet[:, mask_resized]
    fet_avg = np.mean(fet_filt, axis=1)
    fet_std = np.std(fet_filt, axis=1)
    print 'fet_avg.shape', fet_avg.shape
    print 'fet_std.shape', fet_std.shape

    return np.concatenate((fet_avg, fet_std))


def spatial_avg_fets(item, img, fetdic, fets_to_trafo, desired_label,
                     prob_fet_name):
    '''Averages all feature maps for the pixels which predicted the
    desired_label'''
    return trafo_fets_with_prob_mask(
        item, img, fetdic, fets_to_trafo, desired_label, prob_fet_name,
        False, None, fet_avg_trafo
    )


def gram_fets(item, img, fetdic, fets_to_trafo, desired_label, prob_fet_name):
    '''Computes the Gram matrix of feature maps for the pixels which predicted the desired_label'''
    return trafo_fets_with_prob_mask(
        item, img, fetdic, fets_to_trafo, desired_label, prob_fet_name,
        False, None, comp_gram
    )


def mean_std_fets(item, img, fetdic, fets_to_trafo, desired_label,
                  prob_fet_name):
    '''Computes the mean and standard deviation of the features across spatial
    location for the pixels which predicted the desired_label'''
    return trafo_fets_with_prob_mask(
        item, img, fetdic, fets_to_trafo, desired_label, prob_fet_name,
        comp_mean_std
    )


def trafo_fets_with_prob_mask(item, img, fetdic, fets_to_trafo, desired_label,
                              prob_fet_name, trafo_func):
    '''Transforms feature maps for the pixels which predicted the
    desired_label'''
    # Note: Compute mask, the first axis should be 1! (we fed only one image
    # into the network)
    prob = fetdic[prob_fet_name]
    assert prob.shape[0] == 1
    prob = np.squeeze(prob)

    mask = np.argmax(prob, axis=0) == desired_label

    # Resize mask to the size of the feature maps
    fetdic_final = {}
    for name in fets_to_trafo:
        if name == 'img':
            fet = np.transpose(img, (2, 0, 1))
        else:
            fet = fetdic[name]
            assert fet.shape[0] == 1
            fet = fet[0]

        # Nearest neighbor interpolation
        order = 0
        zoom_factor = (
            float(fet.shape[1]) / mask.shape[0],
            float(fet.shape[2]) / mask.shape[1],
        )
        mask_resized = zoom(mask, zoom=zoom_factor, order=order)
        fetdic_final[name] = trafo_func(fet, mask_resized)

    return fetdic_final
