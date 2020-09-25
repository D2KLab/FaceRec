<template src="./Home.html" ></template>

<script>
import { getTrainingSet, train } from '@/services/face-recognition';

const EXAMPLES = {
  memad: 'http://data.memad.eu/yle/a-studio/8a3a9588e0f58e1e40bfd30198274cb0ce27984e',
  antract: 'http://www.ina.fr/media/AFE86004832',
};

export default {
  name: 'Home',
  data() {
    return {
      video: '',
      people: [],
      projects: [],
      training: false,
    };
  },
  computed: {
    exampleVideo() { return EXAMPLES[this.$store.state.proj]; },
  },
  mounted() {
    getTrainingSet(this.$store.state.proj)
      .then((p) => { this.people = p; });
  },
  methods: {

    startTrain() {
      this.training = true;
      train(this.$store.state.proj)
        .then(() => { this.training = false; });
    },
  },
};
</script>
