import os
from app.core.Chunk import EMBEDDING_DIM
from sdk.library_client import LibraryClient
from dotenv import load_dotenv
import cohere

load_dotenv()

client = cohere.ClientV2(os.getenv("COHERE_API_KEY"))


def get_embedding(texts):
    response = client.embed(texts=texts, model="embed-v4.0", embedding_types=[
                            "float"], output_dimension=EMBEDDING_DIM, input_type="search_query")
    return response.embeddings.float_


def main():
    chunk_text = [
        "Cosmic-scale gravity occasionally acts like an enormous distorting magnifying glass. When a massive object such as a galactic black hole or a dark-matter halo sits between Earth and a more distant light source, its gravity warps spacetime enough to bend and refocus the light, producing arcs, rings, or duplicated images of the background object. Astronomers exploit these \"gravitational lenses\" to weigh invisible matter and to peer deeper into the early universe than any telescope could manage alone. Because the lensing effect depends only on mass, not composition, it has become one of the cleanest ways to map the otherwise elusive distribution of dark matter across cosmic filaments.",
        "Far removed from astronomical wonders, the engineers of ancient Rome proved equally adept at confronting a stubborn challenge: how to move fresh water over long distances without pumps. Their answer -- the aqueduct -- combined precisely graded channels, massive stone arcades, and ingenious siphon systems that could hop valleys or burrow mountains. Rome's Aqua Claudia, for instance, descended barely a few centimeters per kilometer yet delivered enough water daily to supply hundreds of public fountains, baths, and private homes. The durable concrete linings and self-cleaning flow rates they perfected two millennia ago remain case studies in low-maintenance civil engineering.",
        "Meanwhile, in kitchens worldwide, an apparently humble jar of flour and water sustains a thriving microbial ecosystem that bakers call a sourdough starter. Wild strains of Saccharomyces yeasts coexist with Lactobacillus bacteria, feeding on complex carbohydrates and excreting carbon dioxide, ethanol, and lactic acid. The CO2 inflates the dough's gluten network while the acids contribute the bread's trademark tang and extend its shelf life by deterring spoilage organisms. Temperature, refresh interval, and flour choice subtly steer the community's balance, giving each starter -- and the loaves it leavens -- a distinctive flavor fingerprint that commercial baker's yeast cannot replicate.",
    ]

    chunked_on_period = [sentence.strip() for text in chunk_text if text.strip() for sentence in text.strip().split('.') if sentence.strip()]
    embeds = get_embedding(chunked_on_period)
    if embeds is None:
        raise ValueError("Failed to get embeddings from Cohere API.")
    # cast to list
    embeds_list = list(embeds)
    # initialize the vector store client
    vector_store = LibraryClient(base_url="http://localhost:8000")
    lib_id = vector_store.create_library(
        name="Demo Library", metadata={"purpose": "demo"})
    print(f"Created library: {lib_id['id']}")
    chunks = [{
        "embedding": embeds_list[i],
        "metadata": {"text": text, "index": i}
    } for i, text in enumerate(chunked_on_period)]

    vector_store.upsert_chunks(library_id=lib_id['id'], chunks=chunks)

    query = "What specific roles do the wild yeast and lactic-acid bacteria play during sourdough fermentation?"

    query_embedding = get_embedding([query])
    results = vector_store.search(
        library_id=lib_id['id'], query_vector=query_embedding[0], k=4)
    print("Top results:")
    for res in results:
        print("-", res[0]['metadata'], res[1])


if __name__ == "__main__":
    main()
