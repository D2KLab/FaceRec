import AnnotatedVideo from './components/AnnotatedVideo.vue';
import Home from './components/Home.vue';
import Person from './components/Person.vue';

const routes = [{
  path: '/',
  name: 'Home',
  component: Home,
},
{
  path: '/video',
  component: AnnotatedVideo,
},
{
  path: '/person',
  component: Person,
},
];

export default routes;
