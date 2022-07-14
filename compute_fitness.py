import os
import argparse
import tqdm 

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

import torch
from torch.nn import CrossEntropyLoss

def calc_fitness(model, prots, tokenizer, device='cuda:0', model_context_len=1023):
    loss_list = []
    loss_fn = CrossEntropyLoss()
    with torch.no_grad():
        for prot in tqdm.tqdm(prots):
            loss_val = 0
            
            sequence_chunks=[]
            if len(prot) < model_context_len:
                sequence_chunks = [prot]
            else:
                len_target_seq = len(prot)
                num_windows = 1 + int( len_target_seq / model_context_len)
                start=0
                for window_index in range(1, num_windows+1):
                    sequence_chunks.append(prot[start:start+model_context_len])
                    start += model_context_len
            
            for chunk in sequence_chunks:
                for p in [chunk, chunk[::-1]]:
                    ids = torch.tensor([tokenizer.encode(p)]).to(device)
                    input_ids = ids[:, :-1]
                    targets   = ids[:, 1:]
                    
                    logits=model(input_ids).logits
                    loss = loss_fn(target=targets.view(-1), input=logits.view(-1,logits.size(-1)))
                    loss_val += -loss.item()
                
            loss_list += [loss_val]
    return np.array(loss_list)

def get_mutated_sequence(focus_seq, mutant, start_idx=1, AA_vocab="ACDEFGHIKLMNPQRSTVWY"):
    """
    Helper function that mutates an input sequence (focus_seq) via an input mutation triplet (substitutions only).
    Mutation triplet are typically based on 1-indexing: start_idx is used for switching to 0-indexing.
    """
    mutated_seq = list(focus_seq)
    for mutation in mutant.split(":"):
        try:
            from_AA, position, to_AA = mutation[0], int(mutation[1:-1]), mutation[-1]
        except:
            print("Issue with mutant: "+str(mutation))
        relative_position = position - start_idx
        assert (from_AA==focus_seq[relative_position]), "Invalid from_AA or mutant position: "+str(mutation)+" from_AA: "+str(from_AA) + " relative pos: "+str(relative_position) + " focus_seq: "+str(focus_seq)
        assert (to_AA in AA_vocab) , "Mutant to_AA is invalid: "+str(mutation)
        mutated_seq[relative_position] = to_AA
    return "".join(mutated_seq)

def main():
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with RITA models.
    """
    parser = argparse.ArgumentParser(description='RITA scoring')
    parser.add_argument('--RITA_model_name_or_path', default="lightonai/RITA_s", type=str, help='Name of or path to RITA model')
    parser.add_argument('--RITA_tokenizer_name_or_path', default="lightonai/RITA_s", type=str, help='Name of or path to RITA tokenizer')
    parser.add_argument('--DMS_reference_file_path', default='./ProteinGym_reference_file_substitutions.csv', type=str, help='Path of DMS folder')
    parser.add_argument('--DMS_data_folder', type=str, help='Path of DMS folder')
    parser.add_argument('--DMS_index', type=int, help='Index of DMS assay to score in reference file')
    parser.add_argument('--output_scores_folder', default=None, type=str, help='Name of folder to write model scores to')
    parser.add_argument('--indel_mode', action='store_true', help='Whether to score sequences with insertions and deletions')
    parser.add_argument('--performance_file', default='RITA_small.csv', type=str, help='Name of file to write performance to')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.RITA_model_name_or_path,trust_remote_code=True)
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.RITA_tokenizer_name_or_path)

    mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
    list_DMS = mapping_protein_seq_DMS["DMS_id"]
    DMS_id=list_DMS[args.DMS_index]
    print("Computing scores for: {} with RITA: {}".format(DMS_id, args.RITA_model_name_or_path))
    DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
    target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].upper()
    
    DMS_data = pd.read_csv(args.DMS_data_folder + os.sep + DMS_file_name, low_memory=False)
    DMS_data['mutated_sequence'] = DMS_data['mutant'].apply(lambda x: get_mutated_sequence(target_seq, x)) if not args.indel_mode else DMS_data['mutant']

    model_scores = calc_fitness(model=model, prots=np.array(DMS_data['mutated_sequence']), tokenizer=tokenizer)
    
    DMS_data['RITA_score']=model_scores
    scoring_filename = args.output_scores_folder+os.sep+DMS_id+'.csv'
    DMS_data[['mutant','RITA_score','DMS_score']].to_csv(scoring_filename, index=False)
    
    spearman, _ = spearmanr(DMS_data['RITA_score'], DMS_data['DMS_score'])

    if not os.path.exists(args.performance_file) or os.stat(args.performance_file).st_size==0:
        with open(args.performance_file,"w") as performance_file:
            performance_file.write("DMS_id,spearman\n")    
    with open(args.performance_file, "a") as performance_file:
        performance_file.write(",".join([DMS_id,str(spearman)])+"\n")

if __name__ == '__main__':
    main()
