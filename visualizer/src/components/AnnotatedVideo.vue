<template src="./AnnotatedVideo.html" ></template>

<style scoped lang="stylus" src='./AnnotatedVideo.styl'></style>

<style>
.rect {
  position: absolute;
  border: 2px solid yellow;
  transition: all 100ms linear;
}
</style>

<script>
/* eslint no-param-reassign: ["error", { "props": false }] */

import palette from 'google-palette';
import { recognise, getLocator } from '@/services/face-recognition';
import { hhmmss } from '@/utils';
import KgPanel from '@/components/KgPanel.vue';

function adaptDimension(bounding, origW, origH, destW, destH) {
  const rW = destW / origW;
  const rH = destH / origH;
  return {
    x: bounding.x * rW,
    y: bounding.y * rH,
    w: bounding.w * rW,
    h: bounding.h * rH,
  };
}

function between(x, min, max) {
  return x >= min && x <= max;
}

export default {
  name: 'AnnotatedVideo',
  components: { KgPanel },
  data() {
    return {
      url: null,
      confidence: 0.7,
      showunknown: false,
      running: false,
      locator: null,
      results: [],
      boxes: [],
      deselected: [],
    };
  },
  computed: {
    data() {
      return this.results
        .filter((d) => d.confidence >= this.confidence || (this.showunknown && d.unknown));
    },
    classes() {
      const classes = [...new Set(this.data.map((c) => c.name))];

      const colours = palette('mpn65', classes.length);
      // other palettes at http://google.github.io/palette.js/
      return classes.map((c, i) => ({
        label: c,
        selected: !this.deselected.includes(c),
        colour: `#${colours[i]}`,
      }));
    },
    displayKgWidget() {
      if (!this.url) return false;
      const host = new URL(this.url).hostname;
      return ['data.memad.eu', 'www.ina.fr'].includes(host);
    },
  },
  mounted() {
    const url = this.$route.query.v.trim();
    this.url = url;
    getLocator(url)
      .then((d) => { this.locator = d; });

    this.trigService();

    this.$root.$on('toSeg', (x) => {
      const start = x.start.split(':').map((ss) => parseInt(ss, 10));
      const sec = start[0] * 3600 + start[1] * 60 + start[2];
      this.goToSecond(sec);
    });
  },
  methods: {
    onPersonToggle($event, person) {
      if (this.deselected.includes(person)) {
        this.deselected.splice(this.deselected.indexOf(person), 1);
      } else this.deselected.push(person);
    },
    goToSecond(second) {
      this.$refs.video.currentTime = second;
    },
    trigService(disableCache = false) {
      recognise(this.$route.query.v, this.$store.state.proj, disableCache)
        .then((data) => {
          this.results = data.tracks || [];
          this.results = this.results.concat(data.feat_clusters || []);
          this.results = this.results.sort((a, b) => ((a.start_npt > b.start_npt) ? 1 : -1));

          this.running = data.status === 'RUNNING';
          if (this.running) {
            // check for new results every 15 seconds
            setTimeout(this.trigService, 15000);
          }
        });
    },
    saveDimensions() {
      this.$video = this.$refs.video;
      const { videoWidth, videoHeight } = this.$video;

      this.$videoWidth = videoWidth;
      this.$videoHeight = videoHeight;
    },
    updateAnnotations() {
      const { offsetWidth, offsetHeight } = this.$video;
      const frags = this.data
        .filter((d) => between(this.$video.currentTime, d.start_npt, d.end_npt));

      this.boxes.forEach((b) => { b.style.display = 'none'; });
      frags.forEach((frag) => {
        const dim = adaptDimension(frag.bounding, this.$videoWidth, this.$videoHeight,
          offsetWidth, offsetHeight);

        let box = this.boxes.find((b) => b.dataset.class === frag.name);
        if (!box) {
          box = document.createElement('div');
          box.classList.add('rect');
          this.boxes.push(box);
          this.$refs.container.appendChild(box);
        }
        box.style.top = `${dim.y}px`;
        box.style.left = `${dim.x}px`;
        box.style.width = `${dim.w}px`;
        box.style.height = `${dim.h}px`;
        box.style.display = 'block';
        box.style.borderColor = this.classes.find((c) => c.label === frag.name).colour;
        box.dataset.class = frag.classLabel;
      });
    },
  },
  filters: {
    hhmmss(value) {
      if (!value) return '';
      return hhmmss(value);
    },
  },

};
</script>
