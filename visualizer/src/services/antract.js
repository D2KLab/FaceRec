import axios from 'axios';
import config from '../../app.config';

export const SERVER = config.api;

const DATE_REGEX = /T(\d{2}:\d{2}:\d{2}).+/;

export async function get(video) {
  /* eslint-disable no-param-reassign */
  const data = await axios.get(`${SERVER}get_metadata`, {
    params: { video },
  });
  const d = data.data;
  if (d.segments) {
    d.segments.forEach((s) => {
      s.start = s.start.replace(DATE_REGEX, '$1');
      s.end = s.end.replace(DATE_REGEX, '$1');
    });
    d.segments = d.segments.sort((a, b) => ((a.start > b.start) ? 1 : -1));
  }

  return d;
}

export default {
  get,
};
