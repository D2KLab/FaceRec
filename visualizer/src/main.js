import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import { createStore } from 'vuex'
import Oruga from '@oruga-ui/oruga-next'
import { bulmaConfig } from '@oruga-ui/theme-bulma'
import App from './App.vue';
import routes from './routes';
import '@oruga-ui/theme-bulma/dist/bulma.css'
import '@mdi/font/css/materialdesignicons.css'

import config from '../app.config'

const app = createApp(App)


const store = createStore({ // eslint-disable-line no-new
  state() {
    return {
      proj: localStorage.project || '',
    }
  },
  mutations: {
    SET_PROJ(state, value) {
      state.proj = value;
      localStorage.project = value;
    },
  },
});

const router = createRouter({
  history: createWebHistory(config.base),
  routes,
});

app.use(router)
app.use(store)
app.use(Oruga, bulmaConfig);

app.mount('#app')
