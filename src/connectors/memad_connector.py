from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://data.memad.eu/sparql-endpoint")


def get_media(media):
    q = """
        SELECT DISTINCT *
    WHERE {
    VALUES ?media { <%s> } 
    ?programme ebucore:isInstantiatedBy ?media .
    ?media a ebucore:MediaResource;
      ebucore:locator ?locator .
    }
    """ % media
    sparql.setQuery(q)
    # print(q)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()["results"]["bindings"]


def get_programme(media):
    sparql.setQuery("""
        SELECT DISTINCT *
    WHERE {
    VALUES ?programme { <%s> } 
    ?programme ebucore:isInstantiatedBy ?media .
    ?media a ebucore:MediaResource;
      ebucore:locator ?locator .
    }
    """ % media)
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
