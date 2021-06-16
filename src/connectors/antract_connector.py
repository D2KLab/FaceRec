import yaml
import json
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


def get_locator_for(uri):
    # the uri can be a ebucore:MediaResource
    results = get_media(uri)
    if len(results) > 0:
        return results[0]

    # the uri can be a ebucore:TVProgramme
    results = get_notice(uri)
    if len(results) > 0:
        return results[0]

    return None


def get_media(media):
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
    return sparql.query()._convertJSON()["results"]["bindings"]


def get_notice(media):
    query = """%s
    SELECT DISTINCT * WHERE { 
    ?analysis a antract:AntractAnalysis ;
             core:document ?media ;
             core:layer / core:segment ?notice .
    ?notice a ina:NoticeSujet ;
        rdfs:label ?title .

    ?media core:instance ?instance .
    ?instance core:http_url ?locator .
    VALUES ?notice { <%s> }
 }""" % (PREFIXES, media)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query()._convertJSON()["results"]["bindings"]


def get_metadata_for(uri):
    x = get_locator_for(uri)
    media = x['media']['value']
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


def get_segment_for(person):
    q = '''%s
    SELECT DISTINCT ?media SAMPLE(?title) as ?title ?start ?end ?url WHERE {
        ?notice ?prop ?person ;
               rdfs:label %s ;
               core:beginTime ?start ;
               core:endTime ?end .
    
        ?analysis a antract:AntractAnalysis ;
                 core:document ?media ;
                 core:layer / core:segment ?notice .
    
        ?media core:instance / core:http_url ?url .
    
        VALUES ?prop {ina:imageContient ina:aPourParticipant ina:aPourInterprete}
     }
    ''' % (PREFIXES, person)

    sparql.setQuery(q)
    sparql.setReturnFormat(JSON)
    return sparql.query()._convertJSON()["results"]["bindings"]
