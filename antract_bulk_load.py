from SPARQLWrapper import SPARQLWrapper, JSON
from src.utils.uri_utils import uri2video, clean_locator
from src import database, tracker
import time

ENDPOINT = "https://okapi.ina.fr/antract/api/saphir/sparql_search"
sparql = SPARQLWrapper(ENDPOINT)

PREFIXES = """
PREFIX core: <http://www.ina.fr/core#> 
PREFIX ina: <http://www.ina.fr/notice.owl#>
PREFIX antract: <http://www.ina.fr/antract#>
"""


def all_media_with(person):
    query = """%s
SELECT distinct * WHERE { 
    ?notice ina:imageContient | ina:aPourParticipant <%s> ;  # Eisenhower
           rdfs:label ?title ;
           core:beginTime ?start ;
           core:endTime ?end .

    ?analysis a antract:AntractAnalysis ;
             core:document ?media ;
             core:layer / core:segment ?notice .

    ?media core:instance ?instance .
    ?instance core:http_url ?url .
 }""" % (PREFIXES, person)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()["results"]["bindings"]

database.init()

for data in all_media_with('http://www.ina.fr/thesaurus/pp/concept_10128605'):
    media = data['media']['value']
    print(media)

    v = database.get_all_about(media, 'antract')
    tracks = [] if (not v or 'tracks' not in v) else v['tracks']
    need_run = not v or (len(tracks) < 1 and v.get('status') != 'RUNNING')
    if need_run:
        locator, v = uri2video(media)
        v['locator'] = clean_locator(locator)
        database.save_metadata(v)

        database.clean_analysis(locator, 'antract')
        database.save_status(locator, 'antract', 'RUNNING')
        try:
            time1 = time.time()
            tracker.main(locator, project='antract', video_speedup=25, export_frames=True)
            time2 = time.time()
            print('{:s} processed in {:.3f} s'.format(media, (time2 - time1)))
        except RuntimeError:
            database.save_status(locator, 'antract', 'ERROR')
