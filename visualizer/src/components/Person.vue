<template src="./Person.html" ></template>

<style scoped lang="stylus" src='./Person.styl'></style>

<script>
import {
  getTrainingSet, SERVER, getDisabled, getAppearances, setDisabled, crawl,
} from '@/services/face-recognition';

function toggle(collection, item) {
  const idx = collection.indexOf(item);
  if (idx !== -1) {
    collection.splice(idx, 1);
  } else {
    collection.push(item);
  }
}

function arraysIdentical(a, b) {
  let i = a.length;
  if (i !== b.length) return false;
  while (i--) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export default {
  name: 'Person',
  data() {
    return {
      person: '',
      paths: [],
      appearences: [],
      disabled: [],
      disabledOrigin: [],
      crawling: false,
      isBlurred: true,
    };
  },
  computed: {
    disabledCount() {
      if (!this.disabled) return 0;
      if (!this.paths) return 0;
      return this.disabled.filter((x) => this.paths.includes(x)).length;
    },
  },
  mounted() {
    this.person = this.$route.query.p;

    getTrainingSet(this.$store.state.proj)
      .then(this.updatePaths);

    getDisabled(this.$store.state.proj)
      .then((d) => {
        this.disabledOrigin = d.slice();
        this.disabled = d.slice();
      });

    getAppearances(this.person, this.$store.state.proj)
      .then((d) => {
        this.appearences = Array.from(new Set(d.map((s) => s.locator)));
      });
  },
  methods: {
    updatePaths(d) {
      const involved = d
        .find((x) => x.class === this.person);
      if (involved) this.paths = involved.path.map((p) => SERVER + p);
    },
    togglePath(_, p) {
      toggle(this.disabled, p);
    },
    reset() {
      this.disabled = this.disabledOrigin.slice();
    },
    hasChanged() {
      return !arraysIdentical(this.disabled, this.disabledOrigin);
    },
    runCrawler() {
      this.crawling = true;
      crawl(this.person, this.$store.state.proj)
        .then(() => getTrainingSet(this.$store.state.proj))
        .then(this.updatePaths);
    },
    saveChange() {
      setDisabled(this.$store.state.proj, this.disabled);
      this.disabledOrigin = this.disabled.slice();
    },
  },
};
</script>
