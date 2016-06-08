import os

import numpy as np
from scipy.ndimage.interpolation import zoom

from cnntools.utils import add_caffe_matclass_to_path


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
                  prob_fet_name, use_crf, crf_params):
    '''Computes the mean and standard deviation of the features across spatial
    location for the pixels which predicted the desired_label'''
    return trafo_fets_with_prob_mask(
        item, img, fetdic, fets_to_trafo, desired_label, prob_fet_name,
        use_crf, crf_params, comp_mean_std
    )


def crf_prob(item, img, fetdic, fets_to_trafo, prob_fet_name, crf_params):
    prob = fetdic[prob_fet_name]
    assert prob.shape[0] == 1
    prob = np.squeeze(prob)

    labels_crf = apply_crf(item, img, prob, crf_params)

    return {
        prob_fet_name: fetdic[prob_fet_name],
        'img': labels_crf,
    }


def trafo_fets_with_prob_mask(item, img, fetdic, fets_to_trafo, desired_label,
                              prob_fet_name, use_crf, crf_params, trafo_func):
    '''Transforms feature maps for the pixels which predicted the
    desired_label'''
    # Note: Compute mask, the first axis should be 1! (we fed only one image
    # into the network)
    prob = fetdic[prob_fet_name]
    assert prob.shape[0] == 1
    prob = np.squeeze(prob)

    # "Clean up" the probability predictions before we use them for masking
    # with a CRF
    if use_crf:
        labels_crf = apply_crf(item, img, prob, crf_params)
        mask = labels_crf == desired_label
    else:
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


def apply_crf(item, img, prob, crf_params):
    '''Applies CRF with the specified crf_params on the probability map given
    the image'''
    if not crf_params:
        raise ValueError('crf_params has to be specified!')

    print 'img.shape', img.shape
    print 'prob.shape', prob.shape
    h, w = img.shape[:2]
    label_count, prob_height, prob_width = prob.shape
    if 'ignore_labels' in crf_params:
        # remove ignored classes, so the CRF won't predict them
        prob[crf_params['ignore_labels']] = 0.0

    zoom_factor = (
        1,
        float(h) / prob_height,
        float(w) / prob_width,
    )
    # Bilinear interpolation
    prob_resized = zoom(prob, zoom=zoom_factor, order=1)

    add_caffe_matclass_to_path()
    from general_densecrf import densecrf_map

    # TODO: Now we are using the MINC CRF, we should generalize it to any
    # densecrf!
    labels_crf = densecrf_map(img, prob_resized.copy(), crf_params)
    #save_crf_predictions(item, img, prob_resized, labels_crf)

    return labels_crf


def save_crf_predictions(item, img, prob_resized, labels_crf):
    from scipy.misc import imsave
    from matclass import dataset, accuracy
    from common.utils import ensuredir
    out_dir = 'crf-debug'
    ensuredir(out_dir)

    for l in range(prob_resized.shape[0]):
        #if l != dataset.NAME_TO_NETCAT['wood']:
            #continue
        img_mask = prob_resized[l, :, :][:, :, np.newaxis]
        prob_img = np.array([1, 0, 0])[np.newaxis, np.newaxis, :] * img_mask
        new_img = prob_img * 0.5 + img * 0.5
        imsave(
            os.path.join(out_dir, '%s-prob-%s-crf.jpg' % (item.id, dataset.NETCAT_TO_NAME[l])),
            new_img
        )
    imsave(
        os.path.join(out_dir, '%s-labels-crf.jpg' % item.id),
        accuracy.labels_to_color(labels_crf)
    )
