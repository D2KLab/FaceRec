<template>
  <div>
    <div class="video-container" ref="container">
      <video ref="video"
      v-bind:src="locator"
      v-on:loadedmetadata="saveDimensions()"
      v-on:timeupdate="updateAnnotations()"
      controls autoplay></video>
    </div>
    <div class="container">

    <div class="columns is-mobile">
      <div class="column legenda">
        <p>In this video:</p>
        <ul id="legenda">
          <li v-for="item in classes" v-bind:key="item.label">
            <span class="square" v-bind:style = "{borderColor: item.colour}"></span>
            {{ item.label }}
          </li>
        </ul>
      </div>
      <div class="column is-two-thirds">
        <ul class="listing">
          <li v-for="d in data" v-bind:key="d.start_npt + d.name">
            <a v-on:click="goToSecond(d.start_npt)">
              <strong>{{d.start_npt}} - {{d.end_npt}}</strong></a>
            {{d.name}} <small>Confidence: {{d.confidence | formatNumber}}</small>
          </li>
        </ul>
      </div>
    </div>
  </div>

  </div>
</template>

<script>
/* eslint no-param-reassign: ["error", { "props": false }] */

import palette from 'google-palette';
import { recognise, getLocator } from '@/face-recognition-service';

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
  data() {
    return {
      locator: null,
      rectStyle: {
        top: 0,
        left: 0,
        height: 0,
        width: 0,
      },
      classes: [],
      boxes: [],
    };
  },
  mounted() {
    getLocator(this.$route.query.v)
      .then((d) => { this.locator = d; });
    recognise(this.$route.query.v)
      .then((data) => {
        this.data = data.results;
        this.data = this.data.sort((a, b) => ((a.start_npt > b.start_npt) ? 1 : -1));
        const classes = [...new Set(this.data.map((c) => c.name))];

        const colours = palette('mpn65', classes.length);
        // other palettes at http://google.github.io/palette.js/
        this.classes = classes.map((c, i) => ({ label: c, colour: `#${colours[i]}` }));
      });
  },
  methods: {
    goToSecond(second) {
      this.$refs.video.currentTime = second;
    },
    saveDimensions() {
      this.$video = this.$refs.video;
      // const { videoWidth, videoHeight } = this.$video;

      // looks like that the video dimensions are fixed in the software
      this.$videoWidth = 864; // videoWidth;
      this.$videoHeight = 486; // videoHeight;
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
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
video {
  max-width: 100vw;
  max-height: 100vh;
  margin: auto;
}
.video-container {
  position: relative;
  text-align: center;
  margin-bottom: 6px;
}
.square {
  display: inline-block;
  height: 1em;
  width: 1em;
  border: 2px solid yellow;
  vertical-align: sub;
}
.legenda {
  background-color: #eee;
  padding: 1.5em;
}
small {
  color: #ccc;
  font-size: 0.7em;
}
.listing {
  max-height: 20em;
  overflow: scroll;
}
</style>

<style>
.rect {
  position: absolute;
  border: 2px solid yellow;
  transition: all 100ms linear;
}
</style>
