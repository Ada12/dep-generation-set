import os

import numpy as np

import constants
from utils import file_utils


def extract_dep2id(project_items):
    """
    根据原始的 project 的 dependencies 列表数据，得到 dep2id。
    :param project_items: list of dict. [{'Project ID': str, 'Dependency List': [str]}].
    :return: {'dep': id}。
        有三个特殊的单词：START、END、PADDING，之后训练 seq2seq 会用到。
    """
    dep2freq = {}
    for project_item in project_items:
        for dep in project_item['Dependency List']:
            if dep not in dep2freq:
                dep2freq[dep] = 1
            else:
                dep2freq[dep] += 1

    dep_freq_items = sorted(dep2freq.items(), key=lambda x: -x[1])
    dep2id = {
        constants.WORD_PADDING: 0,
        constants.WORD_START: 1,
        constants.WORD_END: 2,
    }
    for dep, freq in dep_freq_items:
        if freq < constants.WORD_FREQUENCY_MIN:
            break
        dep2id[dep] = len(dep2id)

    return dep2id


def filter_project_items_by_deps_num(project_items, dep2id):
    """
    每个 project_item 的只保留在 dep2id 中的依赖。
    只保留过滤后依赖数目至少为 DEPS_NUM_MIN 的 project_item。
    :param project_items: list of dict. [{'Project ID': str, 'Dependency List': [str]}].
    :param dep2id: dep 到 id 的映射。
    :return: 过滤后的 project_items。
    """
    filtered_project_items = []
    for project_item in project_items:
        project_id = project_item['Project ID']
        deps = project_item['Dependency List']
        filtered_deps = [dep for dep in deps if dep in dep2id]
        if len(filtered_deps) >= constants.DEPS_NUM_MIN:
            filtered_project_items.append({
                'project_id': project_id,
                'deps': filtered_deps
            })
    return filtered_project_items


def split_train_test(items):
    indices = np.arange(len(items))
    np.random.shuffle(indices)

    test_right = int(len(items) * constants.SPLIT_TEST_RATE)
    test_items = [items[idx] for idx in indices[: test_right]]
    train_items = [items[idx] for idx in indices[test_right:]]

    return train_items, test_items


def word_seqs2id_seqs(raw_items, dep2id):
    """
    将 dep 转为 id，每个 deps 的 个作为 y，其余作为 x。
    :param raw_items: dep 是字符串表示的原始 project_items。
    :param dep2id: dep 到 id 的映射。
    :return:
    """
    seq_list = []
    for item in raw_items:
        id_list = [dep2id[dep] for dep in item['deps']]
        seq_list.append(id_list)
    return seq_list


def split_source_target_seqs(seqs):
    source_seqs = []
    target_seqs = []
    for seq in seqs:
        np.random.shuffle(seq)
        half = len(seq) // 2
        source_seqs.append(seq[: half])
        target_seqs.append(seq[half:])
    return source_seqs, target_seqs


def main():
    project_items = file_utils.load_json(os.path.join(constants.INPUT_DIR, 'deps.json'))
    print('Raw, project_items, total:', len(project_items))

    dep2id = extract_dep2id(project_items)
    print('Dep, total:', len(dep2id))

    filtered_project_items = filter_project_items_by_deps_num(project_items, dep2id)
    print('Filter by deps num, project_items:', len(filtered_project_items))

    train_items, test_items = split_train_test(filtered_project_items)
    print('Split train items ({}), test items ({})'.format(len(train_items), len(test_items)))

    train_seqs = word_seqs2id_seqs(train_items, dep2id)
    test_seqs = word_seqs2id_seqs(test_items, dep2id)
    test_source_seqs, test_target_seqs = split_source_target_seqs(test_seqs)

    # Write objects to disk.
    file_utils.dump_json(os.path.join(constants.INPUT_DIR, 'dep2id.json'), dep2id)

    file_utils.dump_json(os.path.join(constants.INPUT_DIR, 'train_items.json'), train_items)
    file_utils.dump_json(os.path.join(constants.INPUT_DIR, 'test_items.json'), test_items)

    file_utils.dump_json(os.path.join(constants.INPUT_DIR, 'train_seqs.json'), train_seqs)
    file_utils.dump_json(os.path.join(constants.INPUT_DIR, 'test_source_seqs.json'), test_source_seqs)
    file_utils.dump_json(os.path.join(constants.INPUT_DIR, 'test_target_seqs.json'), test_target_seqs)


if __name__ == '__main__':
    main()
