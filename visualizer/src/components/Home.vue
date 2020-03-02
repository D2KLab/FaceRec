<template src="./Home.html" ></template>

<script>
import { getTrainingSet } from '@/face-recognition-service';

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
