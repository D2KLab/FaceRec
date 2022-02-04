import os
import json
import argparse
import requests

SAVE_DIR = './wikidata_cache'
WIKIDATA_SEARCH = 'https://www.wikidata.org/w/api.php'


class WIKIDATA_RECORD:
    def __init__(self, raw):
        self.labels = raw['labels']
        self.label = self.labels['en']['value']
        self.claims = raw['claims']
        self.types = self.get('P31')

    def has_type(self, type_code):
        return type_code in self.types

    def get(self, prop_code):
        if prop_code not in self.claims:
            return []
        return [x['mainsnak']['datavalue']['value']['id'] for x in self.claims[prop_code] if 'datavalue' in x['mainsnak']]

    def get_literal(self, prop_code):
        if prop_code not in self.claims:
            return []
        return [x['mainsnak']['datavalue']['value'] for x in self.claims[prop_code] if 'datavalue' in x['mainsnak']]

    def get_time(self, prop_code):
        if prop_code not in self.claims:
            return []
        return [x['time'] for x in self.get_literal(prop_code)]

    def get_as_label(self, prop_code):
        return [get(x).label for x in self.get(prop_code)]


def get(code, dest_folder=SAVE_DIR):
    dest = os.path.join(dest_folder, str(code) + '.json')
    os.makedirs(dest_folder, exist_ok=True)
    if os.path.isfile(dest):
        with open(dest, 'r') as f:
            d = json.load(f)
            if 'entities' not in d or code not in d['entities']:
                return None
            return WIKIDATA_RECORD(d['entities'][code])

    params = {
        'action': 'wbgetentities',
        'ids': code,
        'format': 'json',
        'language': 'en',
        'uselang': 'en',
        'type': 'item'
    }
    response = requests.get(WIKIDATA_SEARCH, params=params)
    data = response.json()

    with open(dest, 'w') as f:
        json.dump(data, f, indent=4)

    if 'entities' not in data or code not in data['entities']:
        return None

    return WIKIDATA_RECORD(data['entities'][code])


def _as_list(item):
    if item is None:
        return []
    return item if isinstance(item, list) else [item]


def search(k):
    """
    Search in WIKIDATA by string

    :param k: The string to search.
    :param mtc: P for person, B for organisations, None (default) for both
    :param maximum_records: Maximum number of records to retrieve (Default: 10)
    :param start_record: Start record (Default: 1)
    :return: List of records returned by the API
    """
    if not k:
        raise ValueError('Parameter "k" is required.')

    dest = os.path.join(SAVE_DIR, k + '.json')

    if os.path.isfile(dest):
        with open(dest, 'r') as f:
            data = json.load(f)
    else:
        params = {
            'action': 'wbsearchentities',
            'search': k,
            'format': 'json',
            'language': 'en',
            'uselang': 'en',
            'type': 'item'
        }
        response = requests.get(WIKIDATA_SEARCH, params=params)
        data = response.json()

        with open(dest, 'w') as f:
            json.dump(data, f, indent=4)

    return data['search']


if __name__ == '__main__':
    usage = "python wikidata.py -s 'Charles de Gaulle'"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=usage,
                                     add_help=False)
    parser.add_argument('-c', '--code', required=False, default=None)
    parser.add_argument('-s', '--search', required=False, default=None)
    args = parser.parse_args()

    code = args.code
    searchString = args.search

    if code is None and searchString is None:
        raise Exception('provide a valid ISNI or search string in input.')

    if code is not None:
        r = get(code)
        print(r.has_type('Q51'))
        print(r.get_literal('P569'))
        print(r.get_as_label('P27'))

    elif searchString is not None:
        res = search(searchString)
        print('Found', len(res), 'results matching the query:', str(searchString))
        for x in res:
            print(x['id'])
            get(x['id'])
            print(x)
