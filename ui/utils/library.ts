export type Library = {
    id: string;
    name: string;
    metadata: { [key: string]: string };
    chunks: Array<Chunk>;
    created_at: string;
}

export type LibraryListItem = Pick<Library, 'id' | 'name'>;

export type Chunk = {
    id: string;
    metadata: { [key: string]: string };
    embedding: Array<number>;
}

/**
 * 
 * @returns All libraries in the database.
 */
export async function getAllLibraries(): Promise<Array<LibraryListItem>> {
    return await fetch("http://localhost:8000/library").then(res => res.json());
}