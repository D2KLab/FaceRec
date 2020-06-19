import datetime
import yaml
from enum import Enum
from pymongo import MongoClient


on = False
db = None


def init(conf="config/config.yaml"):
    global db, on
    on = True

    with open(conf, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)['mongo']

    client = MongoClient(cfg['server'], int(cfg['port']))
    db = client.facerec

    clean_invalid_states()  # do it at startup


def is_on():
    return on


def now():
    return datetime.datetime.now().isoformat()


class Status(Enum):
    UNKNOWN = 0
    RUNNING = 1
    COMPLETE = 2
    ERROR = 3


def save_metadata(metadata):
    locator = metadata['locator']
    return db.metadata.replace_one({'locator': locator}, metadata, upsert=True)


def get_metadata(uri):
    return db.metadata.find_one({
        '$or': [
            {'media': uri},
            {'programme': uri},
            {'locator': uri},
        ]
    })


# invalidate RUNNING statuses at startup
def clean_invalid_states():
    db.status.remove({'status': Status.RUNNING.value})


def save_status(uri, project, status):
    update = {
        'locator': uri,
        'project': project,
        'status': Status[status].value,
        'timestamp': now()
    }

    return db.status.replace_one({'locator': uri, 'project': project}, update, upsert=True)


def get_status(uri, project):
    s = db.status.find_one({'locator': uri, 'project': project})
    if s is None:
        return None
    return Status(s.get('status', 0))


def clean_analysis(uri, project):
    return db.track.remove({'locator': uri, 'project': project})


def insert_partial_analysis(track):
    return db.track.insert_one(track)


def get_analysis(uri, project):
    return list(db.track.find({'locator': uri, 'project': project}))


def get_all_about(uri, project):
    v = get_metadata(uri)
    if v:
        locator = v['locator']
        status = get_status(locator, project)

        if status and status != Status.ERROR:
            v['status'] = status.name
            v['project'] = project
            v['tracks'] = get_analysis(locator, project)

    return v
