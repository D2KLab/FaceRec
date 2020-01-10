from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://data.memad.eu/sparql-endpoint")


def get_media(media):
    sparql.setQuery("""
        SELECT DISTINCT *
    WHERE {
    VALUES ?media { <http://data.memad.eu/media/8a3a9588e0f58e1e40bfd30198274cb0ce27984e> } 
    ?programme ebucore:isInstantiatedBy ?media .
    ?media a ebucore:MediaResource;
      ebucore:locator ?locator .
    }
    """)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()["results"]["bindings"]


def get_programme(media):
    sparql.setQuery("""
        SELECT DISTINCT *
    WHERE {
    VALUES ?programme { <http://data.memad.eu/media/8a3a9588e0f58e1e40bfd30198274cb0ce27984e> } 
    ?programme ebucore:isInstantiatedBy ?media .
    ?media a ebucore:MediaResource;
      ebucore:locator ?locator .
    }
    """)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()["results"]["bindings"]


def get_locator_for(uri):
    # the uri can be a ebucore:MediaResource
    results = get_media(uri)
    if len(results) > 0:
        return results[0]

    # the uri can be a ebucore:TVProgramme
    results = get_programme(uri)
    if len(results) > 0:
        return results[0]

    return None

