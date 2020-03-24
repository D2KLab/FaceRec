import AnnotatedVideo from './components/AnnotatedVideo.vue';
import Home from './components/Home.vue';
import Person from './components/Person.vue';

const routes = [{
  path: '/',
  name: 'Home',
  component: Home,
},
{
  name: 'video',
  path: '/video',
  component: AnnotatedVideo,
},
{
  name: 'person',
  path: '/person',
  component: Person,
},
];

export default routes;
