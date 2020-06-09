import os
import re
from src.connectors import limecraft_connector as limecraft
from src.connectors import memad_connector as memad
from src.connectors import antract_connector as antract


def uri2video(uri):
    if uri.endswith('.avi') or uri.endswith('.mp4'):
        return uri, {'locator': uri}  # it is already a location
    if uri.startswith('http://data.memad.eu'):
        data = memad.get_locator_for(uri)
        metadata = {}
        for attr, value in data.items():
            metadata[attr] = value['value']

        video_path = limecraft.locator2video(metadata['locator'])
        metadata['locator'] = clean_locator(video_path)
        return video_path, metadata
    elif uri.startswith('http://www.ina.fr/'):  # antract
        data = antract.get_locator_for(uri)
        metadata = {}
        for attr, value in data.items():
            metadata[attr] = value['value']

        metadata['locator'] = 'https://okapi.ina.fr/antract' + metadata['locator']
        return antract.apply_auth(metadata['locator']), metadata
    else:
        return uri, {'locator': uri}  # it is already a location (probably)


def clean_locator(uri):
    return re.sub(r"\?access_token.+", "", uri)


def normalize_video(video_path):
    if video_path.startswith('http'):  # it is a uri!
        video_path, _ = uri2video(video_path)
    elif not os.path.isfile(video_path):
        raise FileNotFoundError('video not found: %s' % video_path)

    print("--> %s" % video_path)
    return video_path
