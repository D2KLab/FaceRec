<template>
  <div id="app">
    <AnnotatedVideo msg="Welcome to Your Vue.js App"/>
  </div>
</template>

<script>
import AnnotatedVideo from './components/AnnotatedVideo.vue';

export default {
  name: 'app',
  components: {
    AnnotatedVideo,
  },
};
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
