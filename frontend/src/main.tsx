import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/instrument-sans';
import '@fontsource/instrument-serif';
import '@fontsource-variable/geist-mono';
import './styles/tokens.css';
import './styles/base.css';
import './styles/primitives.css';
import './styles/shell.css';
import './styles/models.css';
import './styles/viewport.css';
import './styles/steps.css';
import './styles/assistant.css';
import './styles/home.css';
import './styles/settings.css';
import './styles/capture.css';
import './styles/measure.css';
import './styles/auth.css';
import App from './App';
import { AuthGate } from './shell/AuthScreen';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AuthGate>
      <App />
    </AuthGate>
  </StrictMode>,
);
