<template>
  <div id="app">
    <nav class="navbar is-primary" role="navigation" aria-label="main navigation">
      <div class="navbar-brand">
        <a class="navbar-item" href="/visualizer"><b>FaceRec Visualizer</b></a>

        <a class="navbar-item">
          <form class="field has-addons" action="video">
            <div class="control">
              <input name="v" class="input" type="uri" placeholder="Insert URI of a video" required>
            </div>
            <div class="control">
              <button type="submit" class="button">GO</button>
            </div>
          </form>
        </a>
      </div>

    </nav>

    <router-view></router-view>
  </div>
</template>

<script>
// http://data.memad.eu/yle/a-studio/8a3a9588e0f58e1e40bfd30198274cb0ce27984e
export default {
  name: 'app',
};
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
}
</style>
