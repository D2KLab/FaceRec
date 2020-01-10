import yaml
import requests
from datetime import datetime, timedelta

with open("config.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)['limecraft']

# api documentation : https://platform.limecraft.com/api/documentation/#flatdoc/readme/cases/upload-transcribe.md

token = None
token_exp = datetime.now()


# TODO handle auth errors

def _login():
    global token, token_exp
    url = "https://platform.limecraft.com/api/login"
    payload = {
        'username': cfg['username'],  # username or email as string
        'password': cfg['password'],  # password as string
        'rememberMe': False,  # (default is false), longer expiry time
        'useCookies': False  # (default is true), try to set a cookie
    }
    response = requests.post(url, data=payload)

    data = response.json()
    token = data['token']
    token_exp = datetime.now() + timedelta(hours=1)  # token expires after 1 hour


def _get_token():
    global token, token_exp

    # if the token expiration is in the past, the session is no more valid
    if token_exp < datetime.now():
        _login()  # login again

    return token


def locator2video(locator):
    if locator is None or len(locator) == 0:
        return None

    token = _get_token()

    # moa
    r = requests.get(url=locator, params={'access_token': token})
    data = r.json()
    moi = data['hrefs']['moi']

    # moi
    r = requests.get(url=moi, params={'access_token': token})
    data = r.json()
    videos = [d for d in data if d['mimeType'].startswith('video')]
    if len(videos) == 0:
        return None
    download_link = videos[0]['hrefs']['downloadLink']

    # downloadLink
    r = requests.get(url=download_link, params={'access_token': token})
    return r.text
