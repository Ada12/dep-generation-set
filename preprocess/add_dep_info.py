import os

import constants
from utils import file_utils
from utils.word_utils import lemmatize_word_list


def load_deg_tags(tag_infos, dep2id):
    tag2id = {}
    dep_id2tag_ids = {
        0: [],
        1: [],
        2: []
    }
    for tag_info in tag_infos:
        dep = '{}:{}'.format(tag_info['group_id'], tag_info['artifact_id'])
        # 去除空字符、还原词形.
        tags = [lemmatize_word_list([tag])[0] for tag in tag_info['tags'].split('=') if tag]

        tag_ids = []
        for tag in tags:
            if tag not in tag2id:
                tag2id[tag] = len(tag2id)
            tag_ids.append(tag2id[tag])

        dep_id = dep2id[dep]
        dep_id2tag_ids[dep_id] = tag_ids
    return tag2id, dep_id2tag_ids


def main():
    tag_infos = file_utils.load_csv_data(os.path.join(constants.INPUT_DIR, 'dep_infos.log'))
    dep2id = file_utils.load_json(os.path.join(constants.INPUT_DIR, 'dep2id.json'))
    tag2id, dep_id2tag_ids = load_deg_tags(tag_infos, dep2id)

    file_utils.dump_json(os.path.join(constants.INPUT_DIR, 'tag2id.json'), tag2id)
    file_utils.dump_json(os.path.join(constants.INPUT_DIR, 'dep_id2tag_ids.json'), dep_id2tag_ids)


if __name__ == '__main__':
    main()
