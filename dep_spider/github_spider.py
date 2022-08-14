import base64
import os
from time import sleep

import requests

import constants
from utils.log_utils import LOGGER

FETCH_RATE = 3600 / 5000 / 2

AUTH = [
    {
        'username': 'xinleima',
        'password': '5394a41425ad560775d892bd72fd642423375807'
    },
    {
        'username': '272910663@qq.com',
        'password': 'babd259586e6d70b165f7f61dee90d7ad1eb8fdc'
    }
]

HEADER_JSON = 'application/vnd.github.mercy-preview+json'
HEADER_RAW = 'application/vnd.github.VERSION.raw'

fetch_idx = 0


def fetch(url):
    sleep(FETCH_RATE)
    LOGGER.info('fetch {}'.format(url))
    global fetch_idx
    res = requests.get(url,
                       headers={'Accept': HEADER_JSON},
                       auth=(AUTH[fetch_idx % 2]['username'], AUTH[fetch_idx % 2]['password']))
    fetch_idx += 1
    return res.json()


def base64_to_utf8(base64_str):
    return base64.b64decode(base64_str).decode('utf-8')


class RepositoryResource(object):
    def __init__(self, repo_url):
        self.repo_url = repo_url
        self.sha = self._fetch_sha()

    def _fetch_sha(self):
        res = fetch('{}/git/refs/heads/master'.format(self.repo_url))
        return res['object']['sha']

    def fetch_file(self, filename):
        files_res = fetch('{}/git/trees/{}'.format(self.repo_url, self.sha))
        files = files_res['tree']
        for file in files:
            if file['path'] == filename:
                file_res = fetch(file['url'])
                content = base64_to_utf8(file_res['content'])
                return content
        return None


class RepositoryIterator(object):
    def __init__(self, offset, limit):
        self.offset = offset
        self.limit = limit
        self.items = []

        self.first_star = None
        self.last_star = None
        self.cur_page = 0

    def _build_url(self):
        if self.first_star:
            q = 'language:java+stars:<{}'.format(self.first_star)
        else:
            q = 'language:java'
        return 'https://api.github.com/search/repositories?q={}&sort=stars&order=desc&page={}&per_page=100'.format(q,
                                                                                                                   self.cur_page)

    def _update_once(self):
        self.cur_page += 1
        if self.cur_page > 10:
            self.first_star = self.last_star
            self.last_star = None
            self.cur_page = 1

        url = self._build_url()
        res = fetch(url)
        self.last_star = res['items'][-1]['stargazers_count']
        for item in res['items']:
            self.items.append({
                'url': item['url'],
                'full_name': item['full_name']
            })

    def _try_update(self):
        while self.offset >= len(self.items):
            try:
                self._update_once()
            except Exception as err:
                LOGGER.error('_try_update() error: {}'.format(err))

    def __iter__(self):
        return self

    def __next__(self):
        if self.offset >= self.limit:
            raise StopIteration

        self._try_update()
        res = self.items[self.offset]
        self.offset += 1
        return res


def fetch_poms(root_dir, offset, limit):
    pom_files_num = 0
    repos = RepositoryIterator(offset, limit)
    for i, repo in enumerate(repos):
        LOGGER.info('fetch pom#{}, repo: {}'.format(i + offset, repo))
        try:
            repo_resource = RepositoryResource(repo['url'])
            content = repo_resource.fetch_file('pom.xml')
            if content:
                name = '='.join(repo['full_name'].split('/'))
                with open(os.path.join(root_dir, '{}-{}-pom.xml'.format(i + offset, name)), 'w', encoding='utf-8') as fout:
                    fout.write(content)
                pom_files_num += 1
                LOGGER.info('pom_files_num: {}'.format(pom_files_num))
        except Exception as err:
            LOGGER.error('_fetch_pom error: {}, repo: {}'.format(err, repo))


if __name__ == '__main__':
    fetch_poms(os.path.join(constants.INPUT_DIR, 'poms'), 84, 100000)
