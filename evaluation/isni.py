import os
import re
import json
import argparse
import requests
import xmltodict

SAVE_DIR = './isni_cache'
ISNI_SEARCH = 'http://isni.oclc.org/sru/DB=1.2/'


class ISNIRecord:
    def __init__(self, raw):
        for key in raw:
            setattr(self, key, raw[key])

        self.isni_uri = raw['ISNIAssigned']['isniURI']
        self.metadata = raw['ISNIAssigned']['ISNIMetadata']
        dates = list(set([d['marcDate'] for d in _as_list(self.metadata['identity']['personOrFiction']['personalName'])
                          if 'marcDate' in d]))
        self.dates = dates[0].split('-') if len(dates) > 0 else []
        self._raw_uris = _as_list(
            self.metadata['externalInformation']) if 'externalInformation' in self.metadata else []

    def get_uri(self, scope='isni'):
        if scope == 'isni':
            return self.isni_uri

        uris = [u['URI'] for u in self._raw_uris]
        uris = [re.sub('https?://www.wikidata.org/wiki', 'http://www.wikidata.org/entity', u) for u in uris]

        filtered = [u for u in uris if scope in u]
        return filtered[0] if len(filtered) > 0 else None


def get(code, dest_folder=SAVE_DIR):
    dest = os.path.join(dest_folder, str(code) + '.json')
    os.makedirs(dest_folder, exist_ok=True)
    if os.path.isfile(dest):
        with open(dest, 'r') as f:
            return ISNIRecord(json.load(f)['responseRecord'])

    url = f"https://isni.org/isni/{code}/about.xml"
    response = requests.get(url)
    data = xmltodict.parse(response.text)

    with open(dest, 'w') as f:
        json.dump(data, f, indent=4)

    return ISNIRecord(data['responseRecord'])


def _as_list(item):
    if item is None:
        return []
    return item if isinstance(item, list) else [item]


def search(k, mtc=None, maximum_records=10, start_record=1):
    """
    Search in ISNI by string

    :param k: The string to search.
    :param mtc: P for person, B for organisations, None (default) for both
    :param maximum_records: Maximum number of records to retrieve (Default: 10)
    :param start_record: Start record (Default: 1)
    :return: List of records returned by the API
    """
    if not k:
        raise ValueError('Parameter "k" is required.')

    query = f'pica.nw = "{k}"'
    if mtc is not None:
        query += f' and pica.mtc = "{mtc}"'
    dest = os.path.join(SAVE_DIR, query + '.json')

    if os.path.isfile(dest):
        with open(dest, 'r') as f:
            data = json.load(f)
    else:
        params = {
            'query': query,
            'version': 1.1,
            'operation': 'searchRetrieve',
            'stylesheet': 'http://isni.oclc.org/sru/DB=1.2/?xsl=searchRetrieveResponse',
            'recordSchema': 'isni-b',
            'maximumRecords': maximum_records,
            'startRecord': start_record,
            'recordPacking': 'xml',
            'sortKeys': 'RLV,pica,0,,',
            'x-info-5-mg-requestGroupings': 'none'
        }
        response = requests.get(ISNI_SEARCH, params=params)
        print(response.request.url)
        data = xmltodict.parse(response.text)

        with open(dest, 'w') as f:
            json.dump(data, f, indent=4)

    raw_records = data['srw:searchRetrieveResponse']['srw:records']
    if raw_records is None:
        return []
    records = _as_list(raw_records['srw:record'])

    if len(records) == 0:
        return []

    if records[0]['srw:recordData'] is None:
        raise PermissionError(
            'You have reached the daily limit for public searches of the ISNI database. ISNI member organizations '
            'benefit from additional facilities, such as higher search limits and additional search keys. For '
            'further information please contact info@isni.org.')

    return [ISNIRecord(r['srw:recordData']['responseRecord']) for r in records]


if __name__ == '__main__':
    usage = "python isni.py -i 0000000114448576"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=usage,
                                     add_help=False)
    parser.add_argument('-i', '--isni', required=False, default=None)
    parser.add_argument('-s', '--search', required=False, default=None)
    args = parser.parse_args()

    ISNI = args.isni
    searchString = args.search

    if ISNI is None and searchString is None:
        raise Exception('provide a valid ISNI or search string in input.')

    if ISNI is not None:
        print(get(ISNI).dates)

    elif searchString is not None:
        res = search(searchString)
        print('Found', len(res), 'results matching the query:', str(searchString))
        for x in res:
            print(x.isni_uri)
            print(x.get_uri('wikidata'))
