import yaml
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLTransformer import sparqlTransformer

ENDPOINT = "https://okapi.ina.fr/antract/api/saphir/sparql_search"
sparql = SPARQLWrapper(ENDPOINT)

PREFIXES = """
PREFIX core: <http://www.ina.fr/core#> 
PREFIX ina: <http://www.ina.fr/notice.owl#>
PREFIX antract: <http://www.ina.fr/antract#>
"""

with open("config/config.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)['okapi']


def apply_auth(uri):
    return uri.replace(r"https://", "https://%s:%s@" % (cfg['username'], cfg['password']))


def get_locator_for(media):
    query = """%s
SELECT DISTINCT * WHERE { 
    ?analysis a antract:AntractAnalysis ;
             core:document ?media ;
             core:layer / core:segment ?notice .
    ?notice a ina:NoticeSujet ;
        rdfs:label ?title .

    ?media core:instance ?instance .
    ?instance core:http_url ?locator .
    VALUES ?media { <%s> }
 }""" % (PREFIXES, media)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    if len(results):
        return results[0]
    else:
        return {}


def get_metadata_for(media):
    query = {
        'proto': {
            'id': '?media',
            'title': '?title',
            "date": "?date",
            'segments': {
                'id': '?notice',
                'label': '$rdfs:label$required',
                'start': '$core:beginTime$required',
                'end': '$core:endTime',
            }
        },
        '$where': [
            '''[] a antract:AntractAnalysis ;
                         core:document ?media;
                         core:layer / core:segment ?notice ;
                         core:layer / core:segment ?summary''',
            '''?summary a ina:NoticeSommaire ;
                    rdfs:label ?title ;
                    ina:aPourDateDiffusion ?date''',
            '?notice a ina:NoticeSujet'
        ],
        '$values': {
            'media': media,
        },
        '$prefixes': {
            'core': 'http://www.ina.fr/core#',
            'ina': 'http://www.ina.fr/notice.owl#',
            'antract': 'http://www.ina.fr/antract#',
        }
    }
    return sparqlTransformer(query, {'endpoint': ENDPOINT})
