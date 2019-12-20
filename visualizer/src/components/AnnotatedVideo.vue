<template>
  <div>
    <div class="container" ref="container">
      <video ref="video"
      v-bind:src="getVideoLocator()"
      v-on:loadedmetadata="saveDimensions()"
      v-on:timeupdate="updateAnnotations()"
      controls autoplay></video>
    </div>
    <p>
      In this video:
      <ul id="legenda">
        <li v-for="item in classes" v-bind:key="item.label">
          <span class="square" v-bind:style = "{borderColor: item.colour}"></span>
          {{ item.label }}
        </li>
      </ul>
    </p>

  </div>
</template>

<script>
/* eslint no-param-reassign: ["error", { "props": false }] */

import palette from 'google-palette';
import { recognise } from '@/face-recognition-service';

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

export default {
  name: 'AnnotatedVideo',
  props: {
    msg: String,
  },
  data() {
    return {
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
    recognise(this.$route.params.id)
      .then((data) => {
        this.data = data.results;

        const classes = [...new Set(this.data.map((c) => c.name))];

        const colours = palette('mpn65', classes.length);
        // other palettes at http://google.github.io/palette.js/
        this.classes = classes.map((c, i) => ({ label: c, colour: `#${colours[i]}` }));
      });
  },
  methods: {
    saveDimensions() {
      this.$video = this.$refs.video;
      const { videoWidth, videoHeight } = this.$video;

      // looks like that the video dimensions are fixed in the software
      this.$videoWidth = videoWidth; // 1728; // videoWidth;
      this.$videoHeight = videoHeight; // 972; // videoHeight;
    },
    getVideoLocator() {
      return `http://127.0.0.1:5000/video/${this.$route.params.id.replace(/\//g, '_')}.mp4#t=380`;
    },
    updateAnnotations() {
      const { offsetWidth, offsetHeight } = this.$video;
      console.log(this.$video.currentTime);
      const frags = this.data
        .filter((d) => Math.abs(d.npt - this.$video.currentTime) < 1);
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
}
.container {
  position: relative;
}
.square {
  display: inline-block;
  height: 1em;
  width: 1em;
  border: 2px solid yellow;
  vertical-align: sub;
}
ul {
  list-style: none;
  text-align: left;
  display: inline;
  padding-inline-start: 0;
}

li {
  display: inline-block;
  margin: 2px 0.7em;
}
</style>

<style>
.rect {
  position: absolute;
  border: 2px solid yellow;
  transition: all 100ms linear;
}
</style>
