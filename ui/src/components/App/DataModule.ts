import { getAllLibraries } from "../../../utils/library";

export async function rootLoader() {
	const response = await getAllLibraries();
	return response;
}