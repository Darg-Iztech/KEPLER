import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="path to main data directory")

if __name__=='__main__':
    args = parser.parse_args()

    # Load the citation data from citations_per_paper.json
    with open(os.path.join(args.data_dir, 'id_mapping', 'citations_per_paper.json'), 'r') as file:
        citation_data = json.load(file)

    # Load the entity-to-id mapping from entity2id.json
    with open(os.path.join(args.data_dir, 'id_mapping', 'entity2id.json'), 'r') as file:
        entity2id_mapping = json.load(file)

    # Function to replace entities with their IDs
    def replace_entities_with_ids(citation):
        citation['paper_contexts'] = [entity2id_mapping[context] for context in citation['paper_contexts']]
        citation['citations'] = [entity2id_mapping[citation_id] for citation_id in citation['citations']]
        return citation

    # Apply the function to each entry in the citation data
    updated_citation_data = [replace_entities_with_ids(entry) for entry in citation_data]

    # Save the updated citation data to a new file
    with open(os.path.join(args.data_dir, 'id_mapping', 'citations_per_paper_id.json'), 'w') as file:
        json.dump(updated_citation_data, file, indent=4)

    print("Entities replaced with IDs and saved to citations_per_paper_id.json.")
