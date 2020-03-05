<template src="./KgPanel.html" ></template>
<style scoped lang="stylus" src='./KgPanel.styl'></style>

<script>
/* eslint no-param-reassign: "off" */

import memad from '@/services/memad';

export default {
  name: 'KgPanel',
  data() {
    return {
      data: 0,
    };
  },
  props: ['url'],
  mounted() {
    if (this.url.startsWith('http://data.memad.eu')) {
      memad.get(this.url)
        .then((d) => {
          Object.entries(d).forEach(([i, x]) => { if (!Array.isArray(x)) d[i] = [x]; });
          this.data = d;
          console.log(d);
        });
    }
  },
  filters: {
    kgvalue(value) {
      if (!value) return '';
      if (typeof value !== 'object') return value;

      return `<span class="flag-icon flag-icon-${value.language}"></span>${value.value}`;
    },
  },
};
</script>
