import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def load_mnist_txt(file_path):
    data = np.loadtxt(file_path, delimiter=' ')
    images = data.reshape(-1, 28, 28)
    return images


def display_image(image_matrix):
    plt.imshow(image_matrix, cmap='gray')
    plt.axis('off')
    plt.show()


def save_image(image_matrix, output_path, index):
    image = Image.fromarray(np.uint8(image_matrix), 'L')
    file_name = os.path.join(output_path, f'image_{index}.png')
    image.save(file_name)
    file_name = os.path.join(output_path, f'image_{index}.eps')
    image.save(file_name, format='EPS')


if __name__ == "__main__":
    suffix = '38'
    file_path = 'mnist_features_total_{}.txt'.format(suffix)
    images = load_mnist_txt(file_path)

    cluster_path = "data/ex_mnist_{}.txt".format(suffix)
    clusters = []
    with open(cluster_path, 'r') as file:
        for line in file:
            # 去掉行末的换行符并按空格分隔每行
            cluster = list(map(int, line.strip().split()))
            clusters.append(cluster)

    cluster_dict = dict()
    overlap_nodeset = set()
    for (i, cluster) in enumerate(clusters):
        for node in cluster:
            if node not in cluster_dict:
                cluster_dict[node] = list()
            else:
                overlap_nodeset.add(node)
            cluster_dict[node].append(i)

    label_path = "mnist_labels_total_{}.txt".format(suffix)
    label_dict = dict()
    with open(label_path, 'r') as file:
        for (i, line) in enumerate(file):
            # 去掉行末的换行符并按空格分隔每行
            label = list(map(float, line.strip().split()))[0]
            label_dict[str(i)] = label

    output_path = 'mnist_images_{}'.format(suffix)
    os.makedirs(output_path, exist_ok=True)
    for node in overlap_nodeset:
        save_image(images[node], output_path, node)
