<template>
  <div>
    <div class="container" ref="container">
      <video ref="video" src="video/conan.mp4"
      v-on:loadedmetadata="saveDimensions()"
      v-on:timeupdate="updateAnnotations()"
      controls autoplay></video>
    </div>
    <p>Frame: {{currentFrame}}</p>
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

import axios from 'axios';
import palette from 'google-palette';

function parseResults(res) {
  const text = res.data;
  return text.split('\n')
    .filter(x => x)
    .map((line) => {
      const [x1, y1, x2, y2, temp] = line.split(' ');
      const [classNum, classLabel, frame] = temp.split('.');

      return {
        x: parseInt(x1, 10),
        y: parseInt(y1, 10),
        w: x2 - x1,
        h: y2 - y1,
        classNum,
        classLabel,
        frame: parseInt(frame, 10),
        line,
      };
    });
}

function adaptDimension(dim, origW, origH, destW, destH) {
  const rW = destW / origW;
  const rH = destH / origH;
  return {
    x: dim.x * rW,
    y: dim.y * rH,
    w: dim.w * rW,
    h: dim.h * rH,
  };
}

export default {
  name: 'AnnotatedVideo',
  props: {
    msg: String,
  },
  data() {
    return {
      currentFrame: 0,
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
    axios.get('data/bounding.txt')
      .then(parseResults)
      .then((data) => {
        this.data = data;

        const classes = [...new Set(data.map(c => c.classLabel))];

        const colours = palette('mpn65', classes.length);
        // other palettes at http://google.github.io/palette.js/
        this.classes = classes.map((c, i) => ({ label: c, colour: `#${colours[i]}` }));
      });
  },
  methods: {
    saveDimensions() {
      this.video = this.$refs.video;
      // const { videoWidth, videoHeight } = this.video;

      // looks like that the video dimensions are fixed in the software
      this.videoWidth = 1728; // videoWidth;
      this.videoHeight = 972; // videoHeight;
    },
    updateAnnotations() {
      this.currentFrame = Math.floor(this.video.currentTime * 30);
      const { offsetWidth, offsetHeight } = this.video;

      const frags = this.data.filter(d => d.frame === this.currentFrame);
      this.boxes.forEach((b) => { b.style.display = 'none'; });
      frags.forEach((frag) => {
        const dim = adaptDimension(frag, this.videoWidth, this.videoHeight,
          offsetWidth, offsetHeight);

        let box = this.boxes.find(b => b.dataset.class === frag.classLabel);
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
        box.style.borderColor = this.classes.find(c => c.label === frag.classLabel).colour;
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
