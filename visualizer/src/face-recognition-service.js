import axios from 'axios';

const FR_SERVER = 'http://127.0.0.1:5000/';

export async function recognise(video) {
  const data = await axios.get(`${FR_SERVER}recognise`, {
    params: {
      video,
      speedup: 50,
    },
  });
  return data.data;
}

export default {
  recognise,
};
