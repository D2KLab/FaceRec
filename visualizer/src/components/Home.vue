<template src="./Home.html" ></template>

<script>
import { getTrainingSet } from '@/services/face-recognition';

export default {
  name: 'Home',
  data() {
    return {
      video: '',
      people: [],
    };
  },
  mounted() {
    getTrainingSet()
      .then((d) => { this.people = d; });
  },
};
</script>
