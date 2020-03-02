<template src="./Person.html" ></template>

<style scoped lang="stylus" src='./Person.styl'></style>


<script>
import {
  getTrainingSet, SERVER, getDisabled, setDisabled, crawl,
} from '@/face-recognition-service';

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
      disabled: [],
      disabledOrigin: [],
      crawling: false,
    };
  },
  mounted() {
    this.person = this.$route.query.p;

    getTrainingSet()
      .then(this.updatePaths);

    getDisabled()
      .then((d) => {
        this.disabledOrigin = d.slice();
        this.disabled = d.slice();
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
      crawl(this.person).then(getTrainingSet).then(this.updatePaths);
    },
    saveChange() {
      setDisabled(this.disabled);
      this.disabledOrigin = this.disabled.slice();
    },
  },
};
</script>
