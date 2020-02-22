import datetime
import yaml
from enum import Enum
from pymongo import MongoClient

with open("config/config.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)['mongo']

on = False
db = None


def init():
    global db, on
    on = True

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


def save_status(uri, status):
    update = {
        'locator': uri,
        'status': Status[status].value,
        'timestamp': now()
    }

    return db.status.replace_one({'locator': uri}, update, upsert=True)


def get_status(uri):
    s = db.status.find_one({'locator': uri})
    if s is None:
        return None
    return Status(s.get('status', 0))


def clean_analysis(uri):
    return db.track.remove({'locator': uri})


def insert_partial_analysis(track):
    return db.track.insert_one(track)


def get_analysis(uri):
    return list(db.track.find({'locator': uri}))


def get_all_about(uri):
    v = get_metadata(uri)
    if v:
        locator = v['locator']
        status = get_status(locator)

        if status and status != Status.ERROR:
            v['status'] = status.name
            v['tracks'] = get_analysis(locator)
    return v
