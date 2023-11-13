import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="path to main data directory")
parser.add_argument("--raw_corpus", type=str, help="path to original corpus file including entity ids and descriptions")
parser.add_argument("--train_triplets", type=str, help="path to original training data file")
parser.add_argument("--valid_triplets", type=str, help="path to original validation data file")
parser.add_argument("--test_triplets", type=str, help="path to original test data file")
parser.add_argument("--converted_corpus", type=str, help="path to converted text file")
parser.add_argument("--converted_train_triplets", type=str, help="path to converted training file")
parser.add_argument("--converted_valid_triplets", type=str, help="path to converted validation file")
parser.add_argument("--converted_test_triplets", type=str, help="path to converted test file")

if __name__=='__main__':
    args = parser.parse_args()
    Qid={} # Entity to id (line number in the description file)
    Pid={} # Relation to id
    def getNum(s):
        return int(s[1:])

    if not os.path.exists(args.raw_corpus):
        print(f"Raw corpus does not exist at {args.raw_corpus}")
        exit(1)
    if not os.path.exists(os.path.join(args.data_dir, "corpus")):
        os.makedirs(os.path.join(args.data_dir, "corpus"))

    with open(args.raw_corpus, "r") as fin:
        with open(args.converted_corpus, "w") as fout:
            lines = fin.readlines()
            for idx, line in enumerate(lines):
                data = line.split('\t')
                assert len(data) >= 2
                assert (data[0].startswith('H') or data[0].startswith('T'))
                desc = '\t'.join(data[1:]).strip()
                fout.write(desc+"\n")
                Qid[data[0]] = idx

    def convert_triplets(inFile, outFile):
        print(f"Converting {inFile} to {outFile}")
        if not os.path.exists(inFile):
            print(f"Triplets file does not exist at {inFile}")
            exit(1)
        with open(inFile, "r") as fin:
            with open(outFile, "w") as fout:
                lines = fin.readlines()
                for line in lines:
                    data = line.strip().split('\t')
                    assert len(data) == 3
                    if data[1] not in Pid:
                        Pid[data[1]] = len(Pid)
                    fout.write("%d %d %d\n"%(Qid[data[0]], Pid[data[1]], Qid[data[2]]))

    if args.train_triplets is not None:
        convert_triplets(args.train_triplets, args.converted_train_triplets)
    if args.valid_triplets is not None:
        convert_triplets(args.valid_triplets, args.converted_valid_triplets)
    if args.test_triplets is not None:
        convert_triplets(args.test_triplets, args.converted_test_triplets)

    if len(Qid) > 0 and len(Pid) > 0:
        # Also save the id mappings
        id_mapping_dir = os.path.join(args.data_dir, "id_mapping")
        if not os.path.exists(id_mapping_dir):
            os.makedirs(id_mapping_dir)
        with open(f"{id_mapping_dir}/entity2id.json", "w") as entity2idFile:
            json.dump(Qid, entity2idFile)
        with open(f"{id_mapping_dir}/relation2id.json", "w") as relation2idFile:
            json.dump(Pid, relation2idFile)
