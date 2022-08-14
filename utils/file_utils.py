import csv
import json
import os


def ensure_path_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_csv_data(csv_path, fields=None):
    """根据需要的字段，加载csv文件。

    :param csv_path: str。
        csv文件的路径。
    :param fields: None or List[str]。
        为None时，保留所有字段。否则保留列表中的字段。
    :return: List[Dict[str, str]]。
        返回列表，每个元素是一行数据。
        每个元素为一个Dict，key是字段名，value是字段值。
    """
    with open(csv_path, 'r', encoding='utf-8-sig') as fin:
        reader = csv.DictReader(fin)

        items = []
        for row in reader:
            if fields is None:
                item = row
            else:
                item = {field: row[field] for field in fields}
            items.append(item)

    return items


def group_by(items, fields):
    """根据给定的多个字段作为key，聚合数据行。
    :param items: List[Dict]。
        每个元素是一行数据。
        每个元素为一个Dict，key是字段名，value是字段值。
    :param fields: List[str]。
        要作为key的字段列表。
    :return Dict[tuple, List]。
        key是fields字段值的元组，value是item的列表。
    """
    key2items = {}
    for item in items:
        k = tuple([item[field] for field in fields])
        if k not in key2items:
            key2items[k] = [item]
        else:
            key2items[k].append(item)
    return key2items


def write_csv(items, csv_path):
    fieldnames = items[0].keys()
    with open(csv_path, 'w', encoding='utf-8-sig') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)


def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as fin:
        return json.load(fin)


def dump_json(json_path, obj):
    with open(json_path, 'w', encoding='utf-8') as fout:
        json.dump(obj, fout, indent=4)
