# fetches a jp2 tile from disk, uploads it to a buffer, and uses jp2k_mem to decode it
import glymur
import numpy as np


if __name__ == '__main__':
    fname = 'data/example.jp2'
    orig_img_jp2 = glymur.Jp2k(fname)
    orig_img = orig_img_jp2[:]

    with open(fname, 'r') as img_data:
        img_buffer_jp2 = img_data.read()

    img_buffer_jp2 = np.frombuffer(img_buffer_jp2, dtype=np.uint8)
    img_jp2 = glymur.Jp2kMem(img_buffer_jp2)
    #img_jp2.verbose = True
    img = img_jp2[:]

    assert np.all(img == orig_img)
