import axios from 'axios';

const FR_SERVER = '/';

export async function recognise(video) {
  const data = await axios.get(`${FR_SERVER}track`, {
    params: {
      video,
      speedup: 25,
    },
  });
  return data.data;
}

export async function getLocator(video) {
  if (!video) return '';
  const data = await axios.get(`${FR_SERVER}get_locator`, {
    params: { video },
  });
  return data.data;
}

export default {
  recognise,
  getLocator,
};
