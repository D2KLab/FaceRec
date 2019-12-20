import Vue from 'vue';
import VueRouter from 'vue-router';
import App from './App.vue';
import AnnotatedVideo from './components/AnnotatedVideo.vue';

Vue.config.productionTip = false;
Vue.use(VueRouter);

const routes = [
  {
    path: '/',
    redirect: '/video',
    name: 'Home',
    component: { template: '<router-view/>' },
  },
  { path: '/video/:id(.*)', component: AnnotatedVideo },
];

const router = new VueRouter({
  mode: 'history',
  routes,
});

new Vue({
  router,
  render: (h) => h(App),
}).$mount('#app');
