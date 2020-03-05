import axios from 'axios';
import config from '../../app.config';

export const SERVER = config.api;

export async function recognise(video) {
  const data = await axios.get(`${SERVER}track`, {
    params: {
      video,
      speedup: 25,
    },
  });
  return data.data;
}

export async function getLocator(video) {
  if (!video) return '';
  const data = await axios.get(`${SERVER}get_locator`, {
    params: { video },
  });
  return data.data;
}

export async function getTrainingSet() {
  const data = await axios.get(`${SERVER}training-set`);
  return data.data;
}

export async function setDisabled(list) {
  const data = await axios.post(`${SERVER}disabled`, list);
  return data.data;
}

export async function getDisabled() {
  const data = await axios.get(`${SERVER}disabled`);
  return data.data;
}

export async function crawl(q) {
  const data = await axios.get(`${SERVER}crawler`, { params: { q } });
  return data.data;
}
