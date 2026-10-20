import { createApp } from 'vue';
import App from './App.vue';
import './base.css';
import { createPersistence } from './lib/persistence';

// Hydrate before mounting so controls and image inputs never overwrite saved state.
void createPersistence().then(persistence => createApp(App).use(persistence).mount('#app'));
