import Vue from 'vue';
import VueRouter from 'vue-router';
import Buefy from 'buefy';
import numeral from 'numeral';
import App from './App.vue';
import AnnotatedVideo from './components/AnnotatedVideo.vue';
import 'buefy/dist/buefy.css';

Vue.config.productionTip = false;
Vue.use(VueRouter);
Vue.use(Buefy);


Vue.filter('formatNumber', (value) => numeral(value).format('0.00'));


const routes = [
  {
    path: '/',
    name: 'Home',
  },
  { path: '/video', component: AnnotatedVideo },
];

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes,
});

new Vue({
  router,
  render: (h) => h(App),
}).$mount('#app');
