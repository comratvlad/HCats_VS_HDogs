"""Create raw hipster animals dataset"""

import requests
from project_paths import raw_data_paths


def load_dataset(img_size="small", min_samples=1400):
    """Loading popular images dataset from stocksy.com for tags, which are specified in dataset_paths.

    :param img_size: string, optional (default='small')
        It can be "small" (~150x200) or "large" (~500x700).
    :param min_samples: int, optional (default=1400)
        Minimum number of samples, which you like to load (for each tag).
    :return: list of real samples counts for each tag
    """

    assert img_size in ["small", "large"], "img_size must be 'small' or 'large'"
    assert type(min_samples) == int and min_samples > 0, "min_samples must be positive integer"

    tags = raw_data_paths.keys()
    total_item_count = []

    for tag in tags:
        url = "https://www.stocksy.com/search/query?format=json&filters={%%22text%%22:%%22%s%%22}" % tag + \
            "&sort=popular&page=%d" % 1
        total_item_count.append(requests.get(url).json()["response"]["totalItemCount"])

    assert min(total_item_count) >= min_samples, "min_samples is too big, there are no such many of them"

    item_counts = []
    for tag in tags:
        n = 0
        i = 1
        while n < min_samples:
            url = "https://www.stocksy.com/search/query?format=json&filters={%%22text%%22:%%22%s%%22}" % tag + \
                  "&sort=popular&page=%d" % i
            response = requests.get(url).json()['response']

            for j in range(response['returnedItemCount']):
                if img_size == "small":
                    path = response['items'][j]['thumbUrl']
                else:
                    path = response['items'][j]['variations']['jpgFixedWidthDouble']['url']
                image = requests.get(path)
                with open(raw_data_paths[tag] + '{t}_{i}_{j}.jpg'.format(t=tag, i=i, j=j), 'wb') as img:
                    img.write(image.content)
            n += response["returnedItemCount"]
            i += 1
        item_counts.append(n)

    return item_counts


if __name__ == "__main__":
    load_dataset(min_samples=2000, img_size="large")
