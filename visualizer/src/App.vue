<template>
  <div id="app">
    <b-navbar class="is-primary">
      <template slot="brand">
        <b-navbar-item tag="router-link" :to="{ path: '/' }">
          <b>FaceRec Visualizer</b>
        </b-navbar-item>

        <b-navbar-item>
          <form class="field has-addons" action="video">
            <div class="control">
              <input name="v" class="input" type="uri"
              placeholder="Insert URI of a video" required>
            </div>
            <div class="control">
              <button type="submit" class="button">GO</button>
            </div>
          </form>
        </b-navbar-item>

      </template>

      <template slot="end">

        <b-navbar-item class="has-dropdown" tag="div">
          <b-dropdown class="navbar-link" v-model="proj" aria-role="menu">
              <span slot="trigger">
                  <template>
                      <strong class='mobile-only'>Project:</strong> <span>{{proj}}</span>
                  </template>
              </span>

              <b-dropdown-item :value="p" aria-role="menuitem"
              v-for="p in projects" class="navbar-item"
              :key="p" @click="changeProject(p)">
                    {{p}}
              </b-dropdown-item>
          </b-dropdown>
        </b-navbar-item>

      </template>

    </b-navbar>

    <router-view></router-view>
  </div>
</template>

<script>
import { getProjects } from '@/services/face-recognition';

// http://data.memad.eu/yle/a-studio/8a3a9588e0f58e1e40bfd30198274cb0ce27984e
export default {
  name: 'app',
  data() {
    return {
      projects: [],
    };
  },
  computed: {
    proj() {
      return this.$store.state.proj;
    },
  },
  mounted() {
    getProjects().then((p) => {
      this.projects = p;
      if (!this.proj) this.changeProject(p[0]);
    });
  },
  methods: {
    changeProject(project) {
      this.$store.commit('SET_PROJ', project);
      this.$router.go();
    },
  },
};
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
}

@media screen and (min-width: 1024px){
.mobile-only {
  display:none;
}
.navbar .dropdown-content {
  border-radius: 0;
  border-bottom-left-radius: 6px;
  border-bottom-right-radius: 6px;
  box-shadow: 0 8px 8px rgba(10, 10, 10, 0.1)
}

.navbar .dropdown-menu {
  background-color: white;
  border-bottom-left-radius: 6px;
  border-bottom-right-radius: 6px;
  border-top: 2px solid #dbdbdb;
  border-radius: 0;
  left: 0;
  min-width: 100%;
  position: absolute;
  top: 100%;
  z-index: 20;
  padding:0;
}
.navbar .dropdown.navbar-link{
  padding: 0
}
.navbar .dropdown-trigger {
  height: 100%;
  padding: 0.9rem 2.5rem 0.5rem 0.75rem;
  display:inline-block;
}
}
</style>
