<template src="./KgPanel.html" ></template>
<style scoped lang="stylus" src='./KgPanel.styl'></style>

<script>
/* eslint no-param-reassign: "off" */

import memad from '@/services/memad';
import antract from '@/services/antract';

const services = {
  'www.ina.fr': antract,
  'data.memad.eu': memad,
};
export default {
  name: 'KgPanel',
  data() {
    return {
      data: 0,
    };
  },
  props: ['url'],
  mounted() {
    if (!this.url) return;
    const host = new URL(this.url).hostname;

    const service = services[host];
    if (!service) return;

    service.get(this.url)
      .then((d) => {
        Object.entries(d).forEach(([i, x]) => { if (!Array.isArray(x)) d[i] = [x]; });
        this.data = d;
      });
  },
  filters: {
    kgvalue(value) {
      if (!value) return '';
      if (typeof value !== 'object') return value;

      return `<span class="flag-icon flag-icon-${value.language}"></span>${value.value}`;
    },
  },
  methods: {
    toSeg(seg) {
      this.$root.$emit('toSeg', seg);
    },
  },
};
</script>
