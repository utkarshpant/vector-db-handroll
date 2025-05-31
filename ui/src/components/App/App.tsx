import { useLoaderData } from 'react-router';
import type { rootLoader } from './DataModule';
import type { LibraryListItem } from '../../../utils/library';

function App() {
	const loaderData = useLoaderData<typeof rootLoader>();
	return (
		<div className="p-12">
			<h1 className='font-medium font-sans text-lg tracking-wider'>
				Stack Vector Database
			</h1>
      <h2 className='font-medium text-xl my-2'>Libraries</h2>
			{loaderData.map((lib) => <LibraryItem key={lib.id} library={lib} />)}
		</div>
	);
}

function LibraryItem({ library }: { library: LibraryListItem }) {
	return (
		<div className="px-4 py-4 -mx-4 rounded flex flex-col bg-stone-200">
			<h1 className='text-xs tracking-wide uppercase font-medium text-stone-800'>{library.id}</h1>
			<h2 className='text-lg'>{library.name}</h2>
		</div>
	)
}

export default App;
