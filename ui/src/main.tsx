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
  },
  {
    path: '/library/:libraryId',
    action: async ({ request, params }) => {
      if (request.method === "DELETE") {
        const response = await fetch(`http://localhost:8000/library/${params.libraryId}`, {
          method: 'DELETE',
        });
        if (!response.ok) {
          throw new Error('Failed to delete library');
        }
        return null; // Redirect or handle success as needed
      }
      throw new Error('Invalid request method');
    }
  }
], {
  basename: '/ui',
});

createRoot(document.getElementById('root')!).render(
  <RouterProvider router={router} />
)
