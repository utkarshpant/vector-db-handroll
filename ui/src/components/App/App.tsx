import { useFetcher, useLoaderData } from 'react-router';
import type { rootLoader } from './DataModule';
import type { LibraryListItem } from '../../../utils/library';

function App() {
	const loaderData = useLoaderData<typeof rootLoader>();
	return (
		<div className='p-12 w-full min-h-screen flex flex-col'>
			<h1 className='font-medium font-sans text-lg tracking-wider'>Stack Vector Database</h1>
			<h2 className='font-medium text-xl my-2'>Libraries</h2>
			{loaderData.length > 0 ? (
				<div className='flex flex-col gap-2'>
					{loaderData.map((library: LibraryListItem) => (
						<LibraryItem
							key={library.id}
							library={library}
						/>
					))}
				</div>
			) : (
				<div className='w-full h-full flex-1 rounded bg-stone-200 flex items-center justify-center p-12'>
					<p className='text-sm text-stone-500 w-full text-center'>No libraries found. Try creating some Libraries from the Postman Collection.</p>
				</div>
			)}
		</div>
	);
}

function LibraryItem({ library }: { library: LibraryListItem }) {
	const fetcher = useFetcher();
	return (
		<fetcher.Form
			action={`/library/${library.id}`}
			method='DELETE'
			className='px-4 py-4 -mx-4 rounded flex flex-row items-start gap-4 bg-stone-200'
		>
			<button
				type='submit'
				className='w-min rounded-full min-h-min bg-transparent text-black uppercase text-2xl px-3 cursor-pointer'
			>
				&times;
			</button>
			<div className='flex flex-col w-full justify-between'>
				<div className='flex flex-row justify-between items-center'>
					<span className='flex flex-row gap-1 items-baseline'>
						<h2 className='text-lg'>{library.name}</h2>&nbsp;
						<IndexTypeBadge type={library.index_name} />
					</span>
					<h1 className='text-xs tracking-wide uppercase font-medium text-stone-800'>
						{library.id}
						{library.metadata.created_at
							? `- Created ${new Date(library.metadata.created_at).toLocaleString()}`
							: ''}
					</h1>
				</div>
				<div className='my-2'>
					<h2 className='text-xs tracking-wide uppercase font-medium'>Metadata</h2>
					{Object.keys(library.metadata)
						.filter((meta) => meta != 'created_at')
						.map((key) => (
							<p className='text-sm flex items-baseline'>
								{key} &rarr; {library.metadata[key]}
							</p>
						))}
				</div>
			</div>
		</fetcher.Form>
	);
}

function IndexTypeBadge({ type }: { type: string }) {
	return (
		<span className='text-xs font-medium uppercase bg-stone-300 px-2 py-1 rounded tracking-wide'>{type}</span>
	);
}

export default App;
