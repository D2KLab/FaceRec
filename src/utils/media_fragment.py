# Inspired by https:#github.com/tomayac/Media-Fragments-URI
import re
from datetime import datetime
from urllib.parse import urlparse

NPT_PATTERN = re.compile(r'^((npt:)?((\d+:(\d\d):(\d\d))|((\d\d):(\d\d))|(\d+))(\.\d*)?)?$')
SMPTE_PATTERN = re.compile(r'^(\d+:\d\d:\d\d(:\d\d(\.\d\d)?)?)?$')
SMPTE_PREFIX = re.compile(r'^(smpte(-25|-30|-30-drop)?).*')
SMPTE_START = re.compile(r'^smpte(-25|-30|-30-drop)?:')
WALLCLOCK_PATTERN = re.compile(
    r'^((\d{4})(-(\d{2})(-(\d{2})(T(\d{2}):(\d{2})(:(\d{2})(\.(\d+))?)?(Z|(([-+])(\d{2}):(\d{2})))?)?)?)?)?$')
PIXEL_PATTERN = re.compile(r'^(pixel:)?\d+,\d+,\d+,\d+$')
PERCENT_PATTERN = re.compile(r'^percent:\d+,\d+,\d+,\d+$')
XYWH_PREFIX = re.compile(r'(pixel|percent):')

SEPARATOR = '&'


class MediaFragment:
    # noinspection PyDefaultArgument
    def __init__(self, uri, ignore=list()):
        uri_components = urlparse(uri)
        self.provider = uri_components.netloc
        self.hash = parse_component(uri_components.fragment, ignore)
        self.query = parse_component(uri_components.query, ignore)

    def to_string(self):
        return '\n'.join(['Provider: %s' % self.provider,
                          build_string('Query', self.query),
                          build_string('Hash', self.hash)])

    def t(self):
        if 't' in self.hash:
            return self.hash['t']
        if 't' in self.query:
            return self.query['t']
        return None
        # TODO compute relative hash t-frags of query t-frags


def build_string(name, thing):
    s = '[%s]:\n' % name
    for key, values in thing.items():
        s += '  * %s :\n' % key
        for value in values:
            compact = ['      - %s: %s' % (k, str(v)) for k, v in value]
            s += '    [\n%s   ]\n' % '\n'.join(compact)
    return s


def pad(n):
    return f"{n:02}"


def hms_to_npt(hms):
    hhmmss_raw = hms.lower().replace("s", '').replace("m", ":").replace("h", ":")
    hhmmss_arr = hhmmss_raw.split(":")
    npt = [pad(int(n)) for n in hhmmss_arr]

    return ':'.join(npt)


def check_time_validity(hours, minutes, seconds):
    if hours > 23:
        raise ValueError('Hours > 23')

    if minutes > 59:
        raise ValueError('Minutes > 59')

    if seconds >= 60 and (hours or minutes):
        raise ValueError('Seconds >= 60')


def convert_to_seconds_npt(hhmmss):
    if not hhmmss:
        return None

    # possible cases:
    # 12:34:56.789
    #    34:56.789
    #       56.789
    #       56

    hhmmss = hhmmss.split(':')
    if len(hhmmss) > 3:
        raise ValueError('Unexpected format. Accepted: [[hh:]mm:]ss[.ms]')

    seconds = float(hhmmss[-1])
    minutes = int(hhmmss[-2]) if len(hhmmss) > 1 else 0
    hours = int(hhmmss[0]) if len(hhmmss) == 3 else 0

    check_time_validity(hours, minutes, seconds)
    return hours * 3600 + minutes * 60 + seconds


def convert_to_seconds_smtpe(hhmmssff):
    if not hhmmssff:
        return None

    # possible cases:
    # 12:34:56
    # 12:34:56:78
    # 12:34:56:78.90

    hhmmssff = hhmmssff.split(':')
    if len(hhmmssff) < 3 or len(hhmmssff) > 4:
        raise ValueError('Unexpected format. Accepted: hh:mm:ss:frames[.subframes]')

    hours = int(hhmmssff[0])
    minutes = int(hhmmssff[1])
    seconds = int(hhmmssff[2])
    frames = 0
    subframes = 0

    if len(hhmmssff) == 4:
        if '.' in hhmmssff[3]:
            frames, subframes = [int(i) for i in hhmmssff[3].split('.')]
        else:
            frames = int(hhmmssff[3])

    check_time_validity(hours, minutes, seconds)
    return hours * 3600 + minutes * 60 + seconds + frames * 0.001 + subframes * 0.000001


def check_percent_selection(x, y, w, h):
    # checks for valid percent selections
    for n, v in zip(['x', 'y', 'w', 'h'], [x, y, w, h]):
        if v < 0 or v > 100:
            raise ValueError('The accepted range for %s is 0 <= %s <= 100. Found value: %d' % (n, n, v))


def t_parser(value):
    components = value.split(',')
    if len(components) > 2:
        return False

    start = hms_to_npt(components[0])
    end = hms_to_npt(components[1]) if len(components) == 2 else ''
    if not start and not end:
        return False
    if start and not end and ',' in value:
        return False

    # hours:minutes:seconds.milliseconds
    if NPT_PATTERN.match(start) and NPT_PATTERN.match(end):
        start = start.replace('npt:', '')
        # replace a sole trailing dot, which is legal:
        # npt-sec = 1*DIGIT [ "." *DIGIT ]
        start = re.sub(r'\.$', '', start)
        end = re.sub(r'\.$', '', end)

        start_normalized = convert_to_seconds_npt(start)
        end_normalized = convert_to_seconds_npt(end)

        if start_normalized and end_normalized and start_normalized > end_normalized:
            raise ValueError('Please ensure that start < end.')

        if start_normalized or end_normalized:
            return {
                'value': value,
                'unit': 'npt',
                'start': start,
                'end': end,
                'startNormalized': start_normalized,
                'endNormalized': end_normalized
            }
        else:
            raise ValueError('Please ensure that start or end are legal.')

    # hours:minutes:seconds:frames.further-subdivison-of-frames
    prefix = re.sub(SMPTE_PREFIX, '\1', start)
    start = re.sub(SMPTE_START, '', start)
    if SMPTE_PATTERN.match(start) and SMPTE_PATTERN.match(end):
        # we interpret frames as milliseconds, and further-subdivison-of-frames
        # as microseconds. this allows for relatively easy comparison.
        start_normalized = convert_to_seconds_smtpe(start)
        end_normalized = convert_to_seconds_smtpe(end)
        if start and end and start_normalized > end_normalized:
            raise ValueError('Please ensure that start < end.')

        if start_normalized or end_normalized:
            return {
                'value': value,
                'unit': prefix,
                'start': start,
                'end': end
            }
        else:
            raise ValueError('Please ensure that start or end are legal.')

    start = start.replace('clock:', '')
    if WALLCLOCK_PATTERN.match(start) and WALLCLOCK_PATTERN.match(end):
        # ensure ISO 8601 date conformance.
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if start and end and start_dt > end_dt:
            raise ValueError('Please ensure that start < end.')

        return {
            value: value,
            'unit': 'clock',
            start: start,
            end: end
        }

    raise ValueError('Invalid time dimension.')


def xywh_parser(value):
    values = re.sub(XYWH_PREFIX, '', value).split(',')
    x, y, w, h = [int(i) for i in values[0]]

    unit = None
    if PIXEL_PATTERN.match(value):
        if w <= 0 or h <= 0:
            raise ValueError('w and h cannot be <=0')
        unit = 'pixel'

    elif PERCENT_PATTERN.match(value):
        check_percent_selection(x, y, w, h)
        unit = 'percent'

    if unit is not None:
        return {
            'value': value,
            'unit': unit,
            'x': x,
            'y': y,
            'w': w,
            'h': h
        }
    else:
        raise ValueError('Invalid spatial dimension.')


def track_parser(value):
    return {
        'value': value,
        'name': value
    }


def id_parser(value):
    return {
        'value': value,
        'name': value
    }


def chapter_parser(value):
    return {
        'value': value,
        'chapter': value
    }


dimension_parser = {
    't': t_parser,
    'xywh': xywh_parser,
    'track': track_parser,
    'id': id_parser,
    'chapter': chapter_parser
}


# splits an octet string into allowed key-value pairs
def parse_component(input_string, ignore):
    key_values = {}
    pairs = [x.split('=', maxsplit=1) for x in input_string.split(SEPARATOR)]
    pairs = [pair for pair in pairs if len(pair) == 2]

    for key, value in pairs:
        #  skip disabled keys
        if key in ignore:
            continue

        # only allow keys that are currently supported media fragments dimensions
        if key not in dimension_parser:
            continue
        if not value:
            continue

        parser = dimension_parser[key]
        value = parser(value)

        # keys may appear more than once, thus store all values in an array,
        # the exception being &t
        if key not in key_values[key]:
            key_values[key] = list()

        if key == 't':  # replace
            key_values[key][0] = value
        else:  # add
            key_values[key].append(value)

    return key_values
