import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, help="path to original text file")
parser.add_argument("--train", type=str, help="path to original training data file")
parser.add_argument("--valid", type=str, help="path to original validation data file")
parser.add_argument("--test", type=str, help="path to original test data file")
parser.add_argument("--converted_text", type=str, default="Qdesc.txt", help="path to converted text file")
parser.add_argument("--converted_train", type=str, default="train.txt", help="path to converted training file")
parser.add_argument("--converted_valid", type=str, default="valid.txt", help="path to converted validation file")
parser.add_argument("--converted_test", type=str, default="test.txt", help="path to converted test file")

if __name__=='__main__':
    args = parser.parse_args()
    Qid={}  #Entity to id (line number in the description file)
    Pid={}  #Relation to id
    def getNum(s):
        return int(s[1:])
    with open(args.text, "r") as fin:
        with open(args.converted_text, "w") as fout:
            lines = fin.readlines()
            Cnt=0
            for idx, line in enumerate(lines):
                data = line.split('\t')
                assert len(data) >= 2
                #assert (data[0].startswith('Q') or data[0].startswith('S'))
                desc = '\t'.join(data[1:]).strip()
                if data[0].startswith('Q') and getNum(data[0])>1000:
                    continue
                fout.write(desc+"\n")
                Qid[data[0]] = Cnt#idx
                Cnt+=1
