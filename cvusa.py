import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def get_metadata(fname):
    if 'flickr' in fname:
        user_id, photo_id, lat, lon = fname[:-4].rsplit('/', 1)[1].split('_')
        url = 'https://www.flickr.com/photos/%s/%s' % (user_id, photo_id)
        return lat, lon, user_id, photo_id, url
    elif 'streetview' in fname:
        lat, lon, heading = fname[:-4].rsplit('/', 1)[1].split('_')
        return lat, lon, heading
    return None


def get_aerial(lat, lon, zoom):
    lat_bin = int(float(lat))
    lon_bin = int(float(lon))
    return '%s/%d/%d/%s_%s.jpg' % (zoom, lat_bin, lon_bin, lat, lon)


def show_aerial(data):
    fname, lat, lon = data[0:3]
    image_dir = '%s_aerial/' % fname.split('/', 1)[0]

    g_im = mpimg.imread(fname)
    a_im_14 = mpimg.imread('%s%s' % (image_dir, get_aerial(lat, lon, 14)))
    a_im_16 = mpimg.imread('%s%s' % (image_dir, get_aerial(lat, lon, 16)))
    a_im_18 = mpimg.imread('%s%s' % (image_dir, get_aerial(lat, lon, 18)))

    fig = plt.figure()
    plt.subplot(211)
    plt.axis("off")
    plt.imshow(g_im)
    plt.title('ground')
    plt.subplot(234)
    plt.axis("off")
    plt.imshow(a_im_14)
    plt.title('zoom 14')
    plt.subplot(235)
    plt.axis("off")
    plt.imshow(a_im_16)
    plt.title('zoom 16')
    plt.subplot(236)
    plt.axis("off")
    plt.imshow(a_im_18)
    plt.title('zoom 18')


if __name__ == '__main__':
    with open('/media/leon/SSD_8TB/CVUSA/flickr_images.txt', 'r') as f:
        flickr_data = [(x.strip(),) + get_metadata(x.strip()) for x in f]

    show_aerial(flickr_data[10])
    plt.show()
