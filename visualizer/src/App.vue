<template>
  <div id="app">
    <b-navbar class="is-primary">
      <template slot="brand">
        <b-navbar-item tag="router-link" :to="{ path: '/' }">
          <b>FaceRec Visualizer</b>
        </b-navbar-item>

        <b-navbar-item>
          <form class="field has-addons" :action="$router.resolve('video').href">
            <div class="control">
              <input name="v" class="input" type="url"
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
                      <strong class='mobile-only'>Project:</strong> <span>{{labelFrom(proj)}}</span>
                  </template>
              </span>

              <b-dropdown-item :value="p" aria-role="menuitem"
              v-for="p in projects" class="navbar-item"
              :key="p" @click="changeProject(p)">
                    {{labelFrom(p)}}
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

const projLabel = {
  memad: 'MeMAD',
  antract: 'ANTRACT',
};
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
    if (this.proj
      && this.proj !== 'undefined'
      && this.$route.query.project
      && this.proj !== this.$route.query.project) {
      this.changeProject(this.$route.query.project);
    }
    getProjects().then((p) => {
      this.projects = p;
      if (!this.proj || this.proj === 'undefined') this.changeProject(p[0]);
    });
  },
  methods: {
    changeProject(project) {
      this.$store.commit('SET_PROJ', project);
      this.$router.go();
    },
    labelFrom(project) {
      return projLabel[project] || project;
    },
  },
};
</script>

<style lang="stylus" src='./App.styl'></style>
