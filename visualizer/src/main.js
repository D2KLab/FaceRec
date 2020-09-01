import Vue from 'vue';
import Vuex from 'vuex';
import VueRouter from 'vue-router';
import Buefy from 'buefy';
import moment from 'moment';
import numeral from 'numeral';
import App from './App.vue';
import routes from './routes';
import 'buefy/dist/buefy.css';

Vue.config.productionTip = false;
Vue.use(VueRouter);
Vue.use(Buefy);
Vue.use(Vuex);

Vue.filter('formatDate', (value) => value && moment(String(value)).format('DD/MM/YYYY hh:mm'));

Vue.filter('formatNumber', (value) => numeral(value).format('0.00'));

const store = new Vuex.Store({ // eslint-disable-line no-new
  state: {
    proj: localStorage.project || '',
  },
  mutations: {
    SET_PROJ(state, value) {
      state.proj = value;
      localStorage.project = value;
    },
  },
});

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes,
});

new Vue({
  router,
  store,
  render: (h) => h(App),
}).$mount('#app');
