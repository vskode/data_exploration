# Copyright (c) Max Planck Institute of Animal Behavior
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# author: Julian Schaefer-Zimmermann

import torch

def pad_left_right(signal, time, right_pad=False, length_given=False):
    if length_given:
        time_len = time
    else:
        time_len = time.size(0)

    if signal.size(0) == time_len:
        return signal
    if right_pad:
        size_diff = time_len - signal.size(0)
        padded_signal = torch.nn.functional.pad(signal, (0, size_diff), "constant", 0)[:len(time)]
    else:
        size_diff = np.ceil((time_len - signal.size(0)) / 2).astype(int)
        padded_signal = torch.nn.functional.pad(signal, (size_diff, size_diff), "constant", 0)[:len(time)]
    return padded_signal

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunk_and_normalize(data, segment_length=10, sample_rate=8000, normalize=True, max_batch_size=16):
    """
    This is a helper function that chunks an input array into segment_length long chunks
    and (optionally) normalizes each chunk to zero mean and unit variance.
    """
    data = data.squeeze()
    assert data.ndim == 1
    seq_len = round(segment_length * sample_rate)
    if len(data) > seq_len:
        # we need to split the input file into smaller segments
        batched_wav = list(data.split(seq_len))
        # The last segment will have a different length than the others. We right pad with zero
        batched_wav[-1] = pad_left_right(batched_wav[-1], batched_wav[0], right_pad=True)
        # If the batched wav file is longer then our max batch_size, then chunk it
        if len(batched_wav) > max_batch_size:
            batched_wav = list(chunks(batched_wav, max_batch_size))
        else:
            # place in list such that it is a single batch when passed to model
            batched_wav = [batched_wav]
    else:
        batched_wav = [data]

    if normalize:
        b_ = []
        for batch in batched_wav:
            if not torch.is_tensor(batch):
                batch = torch.stack(batch)  # stack the list of tensors
            elif batch.dim() == 1:  # split segments or single segment
                batch = batch.view(1, -1)
            b_.append([torch.nn.functional.layer_norm(x, x.shape).squeeze() for x in batch])
        batched_wav = b_
    return batched_wav
