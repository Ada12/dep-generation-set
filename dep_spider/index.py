import os

import requests
from time import sleep

import numpy as np
from bs4 import BeautifulSoup

import constants
from utils import file_utils

BASE_URL = 'https://mvnrepository.com/artifact'


def extract_info(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    desc_div = soup.find('div', {'class': 'im-description'})
    desc = desc_div.text.replace('\n', '')

    tag_divs = soup.findAll('a', {'class': 'b tag'})
    tags = [tag_div.text for tag_div in tag_divs]

    return desc, tags


def fetch_dep_info(group_id, artificat_id):
    sleep(np.random.rand() + 1.0)
    url = '{}/{}/{}'.format(BASE_URL, group_id, artificat_id)
    print('Fetch', url)
    res = requests.get(url)
    try:
        return extract_info(res.text)
    except Exception as err:
        print('Fetch or extract dep info failed:', group_id, artificat_id, err)
        return None, None


def main():
    dep_info_path = os.path.join(constants.INPUT_DIR, 'dep_infos.log')
    dep_infos = file_utils.load_csv_data(dep_info_path, ['group_id', 'artifact_id'])
    fetched_deps = {(dep_info['group_id'], dep_info['artifact_id']) for dep_info in dep_infos}

    dep2id = file_utils.load_json(os.path.join(constants.INPUT_DIR, 'dep2id.json'))

    with open(dep_info_path, 'a', encoding='utf-8') as fout:
        with open(os.path.join(constants.INPUT_DIR, 'dep_infos_error.log'), 'a', encoding='utf-8') as error_fout:
            for dep in dep2id.keys():
                params = dep.split(':')
                if len(params) != 2:
                    print('Cannot split dep by <:>:', dep)
                    continue
                if (params[0], params[1]) in fetched_deps:
                    print('Already fetch :', params[0], params[1])
                    continue

                desc, tags = fetch_dep_info(params[0], params[1])
                if desc is not None:
                    items = [params[0], params[1], desc, '='.join(tags)]
                    items = ['"{}"'.format(item) for item in items]
                    fout.write(','.join(items) + '\n')
                    fout.flush()
                else:
                    error_fout.write('{} {}\n'.format(params[0], params[1]))
                    error_fout.flush()


if __name__ == '__main__':
    main()
