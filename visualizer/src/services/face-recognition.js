import axios from 'axios';
import config from '../../app.config';

export const SERVER = config.api;

export async function recognise(video, project) {
  const data = await axios.get(`${SERVER}track`, {
    params: {
      video,
      project,
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

export async function getTrainingSet(project) {
  if (!project) return [];
  const data = await axios.get(`${SERVER}training-set`,
    { params: { project } });
  return data.data;
}

export async function getProjects() {
  const data = await axios.get(`${SERVER}projects`);
  return data.data;
}

export async function setDisabled(project, list) {
  const data = await axios.post(`${SERVER}disabled/${project}`, list);
  return data.data;
}

export async function getAppearances(person, project) {
  const data = await axios.get(`${SERVER}appearance/${person}?project=${project}`);
  return data.data;
}

export async function getDisabled(project) {
  const data = await axios.get(`${SERVER}disabled/${project}`);
  const x = data.data;
  return x.map((y) => y.replace('data/', SERVER));
}

export async function crawl(q, project) {
  const data = await axios.get(`${SERVER}crawler`, { params: { q, project } });
  return data.data;
}

export async function train(project) {
  const data = await axios.get(`${SERVER}train/${project}`);
  return data.data;
}
