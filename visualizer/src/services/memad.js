import axios from 'axios';

const API = 'http://grlc.eurecom.fr/api/MeMAD-project/api/program_detail';

async function get(url) {
  const data = await axios.get(API, {
    params: { program: url },
  });
  return data.data[0];
}

export default {
  get,
};
