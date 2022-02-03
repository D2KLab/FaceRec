<template>
  <div id="app">
    <nav class="navbar is-primary" role="navigation" aria-label="main navigation">
      <div class="navbar-brand">
        <router-link class="navbar-item" tag="router-link" :to="{ path: '/' }">
          <b>FaceRec Visualizer</b>
        </router-link>

        <div class="navbar-item">
          <form class="field has-addons" :action="$router.resolve('video').href">
            <div class="control">
              <input name="v" class="input" type="url"
              placeholder="Insert URI of a video" required>
            </div>
            <div class="control">
              <button type="submit" class="button">GO</button>
            </div>
          </form>
        </div>

        <a role="button" class="navbar-burger" aria-label="menu" aria-expanded="false" data-target="nav-menu">
          <span aria-hidden="true"></span>
          <span aria-hidden="true"></span>
          <span aria-hidden="true"></span>
        </a>

      </div>

      <div id="nav-menu" class="navbar-menu">

        <div class="navbar-end">

            <o-dropdown v-model="proj" aria-role="list" rootClass="is-right">
              <template variant="info" #trigger="{ active }">
                <strong class='mobile-only'>Project:</strong> <span>{{labelFrom(proj)}}</span>
                <o-icon customSize="mdi-18px" :icon="active ? 'chevron-up' : 'chevron-down'"></o-icon>
              </template>

              <o-dropdown-item class="navbar-item" :value="p" aria-role="listitem" v-for="p in projects" :key="p" @click="changeProject(p)">
              {{labelFrom(p)}}
            </o-dropdown-item>
          </o-dropdown>

      </div>
    </div>

  </nav>

  <router-view></router-view>
</div>
</template>

<script>
import { getProjects } from '@/services/face-recognition';

const projLabel = {
  memad: 'MeMAD',
  memad_gt: 'MeMAD Ground Truth',
  antract: 'ANTRACT',
};
// http://data.memad.eu/yle/a-studio/8a3a9588e0f58e1e40bfd30198274cb0ce27984e
export default {
  name: 'app',
  data() {
    return {
      projects: [],
      active: false
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
        console.info('Setting project:', project)
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
