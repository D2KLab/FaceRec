<template src="./Home.html" ></template>

<script>
import { getTrainingSet } from '@/services/face-recognition';

export default {
  name: 'Home',
  data() {
    return {
      video: '',
      people: [],
      projects: [],
    };
  },
  mounted() {
    getTrainingSet(this.$store.state.proj)
      .then((p) => { this.people = p; });
  },
};
</script>
