from uuid import uuid5 as UUID, NAMESPACE_DNS
from rdflib import Graph, URIRef, Literal, Namespace, RDF
from rdflib.namespace import DCTERMS, XSD

EBUCORE = Namespace('http://www.ebu.ch/metadata/ontologies/ebucore/ebucore#')
NIF = Namespace('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#')
MEMAD = Namespace('https://memad.eu/ontology#')
OA = Namespace('https://www.w3.org/ns/oa#')
a = RDF['type']

EURECOM = URIRef('http://data.memad.eu/organization/EURECOM')
EURECOM_FACEREC = Literal('EURECOM Face Recognition')


def init_graph():
    g = Graph()

    # set prefixes
    g.namespace_manager.bind('ebucore', EBUCORE)
    g.namespace_manager.bind('nif', NIF)
    g.namespace_manager.bind('memad', MEMAD)
    g.namespace_manager.bind('oa', OA)
    g.namespace_manager.bind('dcterms', DCTERMS)

    return g


def semantify(res):
    data = res['tracks']
    video = res['locator']

    g = init_graph()

    if 'media' in res:
        media = res['media']
        program = res['media']

        video = URIRef(media)
        programme = URIRef(program)

        g.add((programme, EBUCORE['isInstantiatedBy'], video))
    else:
        video = URIRef('http://example.org/' + video)

    for d in data:
        print(d)
        npt = '%.2f' % d['start_npt']
        if 'end_npt' in d:
            npt += ',%.2f' % d['end_npt']
        xywh = d['bounding']['xywh']

        # uuid = UUID('%s%f%d%d%d%d' % (video_id, npt, x, y, w, h))
        frag_uri = video + '#t=npt:%s&xywh=%s' % (npt, xywh)
        frag = URIRef(frag_uri)
        uuid = UUID(NAMESPACE_DNS, frag_uri)
        body = URIRef('http://data.memad.eu/person-identification/%s' % uuid)

        g.add((frag, a, EBUCORE['MediaFragment']))
        g.add((frag, EBUCORE['isMediaFragmentOf'], video))

        g.add((body, a, NIF['Annotation']))
        g.add((body, a, MEMAD['VisualPersonIdentification']))
        g.add((body, RDF['value'], Literal(d['name'])))
        g.add((body, NIF['taIdentProv'], EURECOM_FACEREC))
        g.add((body, NIF['taIdentConf'], Literal(d['confidence'], datatype=XSD['decimal'])))

        annotation = URIRef('http://data.memad.eu/annotation/video-annotation/%s' % uuid)

        g.add((annotation, a, OA['Annotation']))
        g.add((annotation, DCTERMS['creator'], EURECOM))
        g.add((annotation, DCTERMS['created'], Literal(data['timestamp'], datatype=XSD['datetime'])))
        g.add((annotation, DCTERMS['motivatedBy'], OA['identifying']))
        g.add((annotation, OA['hasTarget'], frag))
        g.add((annotation, OA['hasBody'], body))

    return g.serialize(format="turtle")
