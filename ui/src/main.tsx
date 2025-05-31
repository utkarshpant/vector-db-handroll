import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router'
import './index.css'
import App from './components/App/App.tsx'
import { rootLoader } from './components/App/DataModule.ts';

const router = createBrowserRouter([
  {
    path: '/',
    Component: App,
    loader: rootLoader,
  }
], {
  basename: '/ui',
});

createRoot(document.getElementById('root')!).render(
  <RouterProvider router={router} />
)
