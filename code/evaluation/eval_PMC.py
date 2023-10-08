import torch 
import numpy as np 
import editdistance
import json
import random
import string
import os
import ir_measures
from ir_measures import *
import time
import ast 

def get_agg_tid(tid_obj) :
    at_ = ['1', '2', '3', '4']
    agg_t = {}
    for t_ in tid_obj :
        _ = t_[:1]
        k = at_.index(_)    
        if k+1 in agg_t :
            agg_t[k+1]+= list(tid_obj[t_])
        else :
            agg_t[k+1]= list(tid_obj[t_])
    return agg_t


def get_binary_score(pans, gans) :
    # print('\n get_binary_score(pans, gans)', pans, gans)
    if pans in ['True', 'Yes', 'yes', 'true', 1, '1'] :
        pans = True   
    elif pans in ['False', 'No', 'no', 'false', 0, '0'] :
        pans = False 
    else : 
        pans == -9999 
    gans = True if gans in ['True', 'Yes', 'yes', 'true'] else False
    # print('get_binary_score(pans, gans)', pans, gans)
    if pans == gans :
        return 1 
    else : 
        return 0


def get_numeric_score(pred, gt, tolerance=0.05):
    # print('In numeric score', pred, gt)
    # error = abs(gt - pred) / (gt+1e-9)
    error = abs(gt - pred) / (gt+1e-9)
    # tolerance = 
    # print('err', error)
    if error <= tolerance:
        return 1 
    else : 
        return 0 
        

def get_string_score(pans, gans) :
    # print('get_string_score(pans)', pans)
    # print('get_string_score(gans)',  gans)
    sc = editdistance.eval(str(pans).lower().strip('\n'), str(gans).lower().strip('\n'))
    if sc == 0 :
        return 1 
    else : 
        return 0

def get_unranked_score(pans, gans) :
    # print('\n get_unranked_score(pans, gans)', pans, '##-##', gans) 
    pan = [str(_).lower() for _ in pans]
    gan = [str(_).lower() for _ in gans]
    # gan = gans[list(gans.keys())[0]]
    print('\n get_unranked_score(pan, gan)', pan, '##-##\n', gan) 
    relevant_items_in_collection = len(gan)
    # print('relevant_items_in_collection',relevant_items_in_collection) 
    items_retrieved = len(pan)
    # print('items_retrieved', items_retrieved) 
    relevant_items_retrieved = 0
    for p in pan :
        for g in gan:  
            sc = editdistance.eval(p, g)
            # print(sc, p, g )
            if sc == 0 :
                relevant_items_retrieved+=1
    # print('relevant_items_retrieved', relevant_items_retrieved) 
    P, R = relevant_items_retrieved/relevant_items_in_collection , relevant_items_retrieved/items_retrieved
    # print('PR', P, R, '\n') 
    # exit()
    return 2*((P*R)/(P+R)) if (P+R) >0 else 0.0

def get_ranked_score(qrels, prels) :
    if len(qrels[list(qrels.keys())[0]]) == 1 :
        # print('in ranked list', qrels, prels)
        return 1 if  qrels[list(qrels.keys())[0]] == prels[list(prels.keys())[0]] else 0 
    else : 
        # print('in ranked list', qrels, prels)
        _ = ir_measures.calc_aggregate([Bpref, nDCG@10], qrels, prels)
        # if _[nDCG@10] < 1 :
            # print('in ranked list', qrels, prels, _)
        return _[nDCG@10]

def get_qrel(pred_obj, gt_obj):
    # print('in get_qrel', pred_obj, gt_obj)
    qrel = {gt_obj['qa_id'] : {}}
    # print(qrel)
    for ix, k in enumerate(gt_obj['answer']) :
        if isinstance(k, list):
            k = [str(_) for _ in k]
            k = "".join(k)
        qrel[gt_obj['qa_id']].update({str(k) : ix})
    # print(qrel)
    prel = {pred_obj['qa_id'] : {}}
    # print(prel)
    for ix, k in enumerate(pred_obj['predicted_answer']) :
        if isinstance(k, list):
            k = [str(_) for _ in k]
            k = "".join(k)
        prel[pred_obj['qa_id']].update({str(k) : ix})
    # print(prel)
    return qrel, prel


def run_eval(pred_dir, gt_dir, sd=None):
    chart_files = os.listdir(gt_dir)
    score_obj = {
        'string_score' : {},
        'num_score' : {},
        'rank_score' : {},
        'urank_score' : {},
        'bi_score' : {}
    }
    print('In main')
    print('pred_dir', pred_dir)
    print('gt_dir :: total chart files',  gt_dir, len(chart_files))
    count_pred = 0
    for cix, chart in enumerate(chart_files) :
        # print('\n \n Working on chart : ', chart)
        gt_js = json.load(open(os.path.join(gt_dir, chart)))
        fl_ = os.path.join(pred_dir, chart)
        # print('pred file', fl_)
        if os.path.isfile(fl_) :
            pr_js = json.load(open(fl_))
            # print('\n pr_js ', pr_js)
            pr_obj = {}
            for pr in pr_js : 
                pr_obj.update({pr['qa_id'] : pr})
        else : 
            pr_obj = None
            count_pred+=1
            continue
        pred_qrel= {}
        gt_qrel= {}
        # print('Total ', len(gt_js), ' qa pairs in GT JSON List')
        # print(gt_js)
        # print('Total ', len(pr_obj), ' qa pairs in Pred JSON List')
        # print(pr_obj)
        # if cix-count_pred > 5 :
            # exit()
        for gt in gt_js :
            gt_id = gt['qa_id']
            gt_ans = gt['answer']
            pans = pr_obj[gt_id]["predicted_answer"] if (pr_obj is not None) and (gt_id in pr_obj) else None

            # pans = pred[i]['predicted_answer']
            # gt_ans = pred[i]['ground_truth_answer']
            if gt['answer_type'] == "Ranked List" :
                if pans is None: 
                    score_obj['rank_score'].update({gt_id : 0})
                else : 
                    gt_qrel, pred_qrel = get_qrel(pr_obj[gt_id], gt)
                    score_obj['rank_score'].update({gt_id :get_ranked_score(gt_qrel, pred_qrel)})
                    # print(gt['question'])
            if gt['answer_type'] == "Unranked List" :
                if pans is None : 
                    score_obj['urank_score'].update({gt_id : 0})
                else : 
                    # gt_qrel, pred_qrel = get_qrel(pr_obj[gt_id], gt)
                    # print('-------p-------', pans)
                    # print('-------g-------', gt_ans[:1])
                    if not isinstance(pans, list):
                        try : 
                            pans = ast.literal_eval(pans)
                            print('-------Unranked List Exception not instance list typecast -------', pans)
                            score_obj['urank_score'].update({gt_id :get_unranked_score(pans, gt_ans)})
                        except : 
                            score_obj['urank_score'].update({gt_id :0})
                    else :
                        score_obj['urank_score'].update({gt_id :get_unranked_score(pans, gt_ans)})
            if gt['answer_type'] == "Numerical" :
                if pans is None : 
                    score_obj['num_score'].update({gt_id : 0})
                else : 
                    try : 
                        pans = float(pans)
                        gt_ans = float(gt_ans)
                        score_obj['num_score'].update({gt_id :get_numeric_score(pans, gt_ans)})
                    except :
                        # print('---', pans, gt_ans)
                        score_obj['num_score'].update({gt_id : 0})
            if gt['answer_type'] == "Binary" :
                if pans is None : 
                    score_obj['bi_score'].update({gt_id : 0})
                else : 
                    score_obj['bi_score'].update({gt_id :get_binary_score(pans, gt_ans)})
            if gt['answer_type'] == "String" :
                if pans is None : 
                    score_obj['string_score'].update({gt_id : 0})
                else : 
                    score_obj['string_score'].update({gt_id :get_string_score(pans, gt_ans)})
        
    # print(score_obj)

    tidobj = np.load('./t_id_map.npy', allow_pickle=True)
    tidobj = tidobj.item()
    agg = get_agg_tid(tidobj)
    
    cobj = np.load('./ctype_id_map.npy', allow_pickle=True)
    cobj = cobj.item()
   


    gtstring = len(score_obj['string_score'])
    predstring =  sum([score_obj['string_score'][_] for _ in score_obj['string_score']])

    gtnum = len(score_obj['num_score'])
    prednum = sum([score_obj['num_score'][_] for _ in score_obj['num_score']])

    gtrank = len(score_obj['rank_score'])
    predrank = sum([score_obj['rank_score'][_] for _ in score_obj['rank_score']])

    gturank = len(score_obj['urank_score'])
    predurank =  sum([score_obj['urank_score'][_] for _ in score_obj['urank_score']])

    gtbi = len(score_obj['bi_score'])
    predbi =  sum([score_obj['bi_score'][_] for _ in score_obj['bi_score']])


    gtStruct = len([_ for _ in agg[1] if _ in score_obj['string_score']]) + \
                len([_ for _ in agg[1] if _ in score_obj['num_score']]) + \
                len([_ for _ in agg[1] if _ in score_obj['rank_score']]) + \
                len([_ for _ in agg[1] if _ in score_obj['urank_score']]) +   \
                len([_ for _ in agg[1] if _ in score_obj['bi_score']]) +\
                len([_ for _ in agg[2] if _ in score_obj['string_score']]) +\
                len([_ for _ in agg[2] if _ in score_obj['num_score']]) +\
                len([_ for _ in agg[2] if _ in score_obj['rank_score']]) + \
                len([_ for _ in agg[2] if _ in score_obj['urank_score']]) +\
                len([_ for _ in agg[2] if _ in score_obj['bi_score']])
    
    predStruct =  sum([score_obj['string_score'][_] for _ in agg[1] if _ in score_obj['string_score']]) +\
                sum([score_obj['num_score'][_] for _ in agg[1] if _ in score_obj['num_score']]) + \
                sum([score_obj['rank_score'][_] for _ in agg[1] if _ in score_obj['rank_score']]) +\
                sum([score_obj['urank_score'][_] for _ in agg[1] if _ in score_obj['urank_score']]) +\
                sum([score_obj['bi_score'][_] for _ in agg[1] if _ in score_obj['bi_score']]) +\
                sum([score_obj['string_score'][_] for _ in agg[2] if _ in score_obj['string_score']]) +\
                sum([score_obj['num_score'][_] for _ in agg[2] if _ in score_obj['num_score']]) +  \
                sum([score_obj['rank_score'][_] for _ in agg[2] if _ in score_obj['rank_score']]) +\
                sum([score_obj['urank_score'][_] for _ in agg[2] if _ in score_obj['urank_score']]) +\
                sum([score_obj['bi_score'][_] for _ in agg[2] if _ in score_obj['bi_score']]) 
                


    gtRet = len([_ for _ in agg[3] if _ in score_obj['string_score']]) + \
            len([_ for _ in agg[3] if _ in score_obj['num_score']]) +\
            len([_ for _ in agg[3] if _ in score_obj['rank_score']]) + \
            len([_ for _ in agg[3] if _ in score_obj['urank_score']]) +\
            len([_ for _ in agg[3] if _ in score_obj['bi_score']]) 
                
    
    predRet =  sum([score_obj['string_score'][_] for _ in agg[3] if _ in score_obj['string_score']]) +\
                sum([score_obj['num_score'][_] for _ in agg[3] if _ in score_obj['num_score']]) +  \
                sum([score_obj['rank_score'][_] for _ in agg[3] if _ in score_obj['rank_score']]) +\
                sum([score_obj['urank_score'][_] for _ in agg[3] if _ in score_obj['urank_score']]) + \
                sum([score_obj['bi_score'][_] for _ in agg[3] if _ in score_obj['bi_score']])  
                 
    gtRes = len([_ for _ in agg[4] if _ in score_obj['string_score']]) + \
            len([_ for _ in agg[4] if _ in score_obj['num_score']]) +\
            len([_ for _ in agg[4] if _ in score_obj['rank_score']]) + \
            len([_ for _ in agg[4] if _ in score_obj['urank_score']]) +   \
            len([_ for _ in agg[4] if _ in score_obj['bi_score']]) 
                
    
    predRes =  sum([score_obj['string_score'][_] for _ in agg[4] if _ in score_obj['string_score']]) +\
                sum([score_obj['num_score'][_] for _ in agg[4] if _ in score_obj['num_score']]) +  \
                sum([score_obj['rank_score'][_] for _ in agg[4] if _ in score_obj['rank_score']]) +\
                sum([score_obj['urank_score'][_] for _ in agg[4] if _ in score_obj['urank_score']]) + \
                sum([score_obj['bi_score'][_] for _ in agg[4] if _ in score_obj['bi_score']])  
                 

    gtLine  = len([_ for _ in cobj['line'] if _ in score_obj['string_score']]) +\
            len([_ for _ in cobj['line'] if _ in score_obj['num_score']]) +\
            len([_ for _ in cobj['line'] if _ in score_obj['rank_score']]) + \
            len([_ for _ in cobj['line'] if _ in score_obj['urank_score']]) +\
            len([_ for _ in cobj['line'] if _ in score_obj['bi_score']]) 
   
    predLine =  sum([score_obj['string_score'][_] for _ in cobj['line'] if _ in score_obj['string_score']]) +\
                sum([score_obj['num_score'][_] for _ in cobj['line'] if _ in score_obj['num_score']]) +  \
                sum([score_obj['rank_score'][_] for _ in cobj['line'] if _ in score_obj['rank_score']]) +\
                sum([score_obj['urank_score'][_] for _ in cobj['line'] if _ in score_obj['urank_score']]) +\
                sum([score_obj['bi_score'][_] for _ in cobj['line'] if _ in score_obj['bi_score']])  


    gtVbar  = len([_ for _ in cobj['vertical bar'] if _ in score_obj['string_score']]) +\
            len([_ for _ in cobj['vertical bar'] if _ in score_obj['num_score']]) +\
            len([_ for _ in cobj['vertical bar'] if _ in score_obj['rank_score']]) + \
            len([_ for _ in cobj['vertical bar'] if _ in score_obj['urank_score']]) +\
            len([_ for _ in cobj['vertical bar'] if _ in score_obj['bi_score']]) 
   
    predVbar =  sum([score_obj['string_score'][_] for _ in cobj['vertical bar'] if _ in score_obj['string_score']]) +\
                sum([score_obj['num_score'][_] for _ in cobj['vertical bar'] if _ in score_obj['num_score']]) +  \
                sum([score_obj['rank_score'][_] for _ in cobj['vertical bar'] if _ in score_obj['rank_score']]) +\
                sum([score_obj['urank_score'][_] for _ in cobj['vertical bar'] if _ in score_obj['urank_score']]) +\
                sum([score_obj['bi_score'][_] for _ in cobj['vertical bar'] if _ in score_obj['bi_score']])  


    gtHbar  = len([_ for _ in cobj['horizontal bar'] if _ in score_obj['string_score']]) +\
            len([_ for _ in cobj['horizontal bar'] if _ in score_obj['num_score']]) +\
            len([_ for _ in cobj['horizontal bar'] if _ in score_obj['rank_score']]) + \
            len([_ for _ in cobj['horizontal bar'] if _ in score_obj['urank_score']]) +\
            len([_ for _ in cobj['horizontal bar'] if _ in score_obj['bi_score']]) 
   
    predHbar =  sum([score_obj['string_score'][_] for _ in cobj['horizontal bar'] if _ in score_obj['string_score']]) +\
                sum([score_obj['num_score'][_] for _ in cobj['horizontal bar'] if _ in score_obj['num_score']]) +  \
                sum([score_obj['rank_score'][_] for _ in cobj['horizontal bar'] if _ in score_obj['rank_score']]) +\
                sum([score_obj['urank_score'][_] for _ in cobj['horizontal bar'] if _ in score_obj['urank_score']]) +\
                sum([score_obj['bi_score'][_] for _ in cobj['horizontal bar'] if _ in score_obj['bi_score']])  


    gtScat  = len([_ for _ in cobj['scatter'] if _ in score_obj['string_score']]) +\
            len([_ for _ in cobj['scatter'] if _ in score_obj['num_score']]) +\
            len([_ for _ in cobj['scatter'] if _ in score_obj['rank_score']]) + \
            len([_ for _ in cobj['scatter'] if _ in score_obj['urank_score']]) +\
            len([_ for _ in cobj['scatter'] if _ in score_obj['bi_score']]) 
   
    predScat =  sum([score_obj['string_score'][_] for _ in cobj['scatter'] if _ in score_obj['string_score']]) +\
                sum([score_obj['num_score'][_] for _ in cobj['scatter'] if _ in score_obj['num_score']]) +  \
                sum([score_obj['rank_score'][_] for _ in cobj['scatter'] if _ in score_obj['rank_score']]) +\
                sum([score_obj['urank_score'][_] for _ in cobj['scatter'] if _ in score_obj['urank_score']]) +\
                sum([score_obj['bi_score'][_] for _ in cobj['scatter'] if _ in score_obj['bi_score']])  


    gtVbox  = len([_ for _ in cobj['line'] if _ in score_obj['string_score']]) +\
            len([_ for _ in cobj['line'] if _ in score_obj['num_score']]) +\
            len([_ for _ in cobj['line'] if _ in score_obj['rank_score']]) + \
            len([_ for _ in cobj['line'] if _ in score_obj['urank_score']]) +\
            len([_ for _ in cobj['line'] if _ in score_obj['bi_score']]) 
   
    predVbox =  sum([score_obj['string_score'][_] for _ in cobj['vertical box'] if _ in score_obj['string_score']]) +\
                sum([score_obj['num_score'][_] for _ in cobj['vertical box'] if _ in score_obj['num_score']]) +\
                sum([score_obj['rank_score'][_] for _ in cobj['vertical box'] if _ in score_obj['rank_score']]) +\
                sum([score_obj['urank_score'][_] for _ in cobj['vertical box'] if _ in score_obj['urank_score']]) +\
                sum([score_obj['bi_score'][_] for _ in cobj['vertical box'] if _ in score_obj['bi_score']])  

    totalgt  = gtstring +  gtnum + gtrank + gturank + gtbi
    totalpred  = predstring + prednum + predrank + predurank + predbi
    print('*'*80)
    print('Exhaustive Set')
    print('Total Evaluated charts ', cix+1-count_pred)
    print('Total Evaluated qa pairs ', totalgt)
    print('Total Accuracy ', totalpred/totalgt)
    print('-'*10)
    print('Total Evaluated String qa pairs ',gtstring )
    print('String Acc ', predstring/gtstring )
    print('-'*10)
    print('Total Evaluated Numeric qa pairs ', gtnum)
    print('Numeric Acc ', prednum/gtnum )
    print('-'*10)
    print('Total Evaluated Ranked qa pairs ', gtrank)
    print('Rank Acc ', predrank/gtrank )
    print('-'*10)
    print('Total Evaluated Unranked qa pairs ', gturank)
    print('Urank Acc ', predurank/gturank )
    print('-'*10)
    print('Total Evaluated Binary qa pairs ', gtbi)
    print('Bi Acc ', predbi/gtbi )
    print('flat - ', totalpred/totalgt, predstring/gtstring, prednum/gtnum, predrank/gtrank, predurank/gturank, predbi/gtbi)
    print('-'*30)
    print('Total Evaluated Structure qa pairs ', gtStruct)
    print('Struc Acc ', predStruct/gtStruct )
    print('-'*10)
    print('Total Evaluated Ret qa pairs ', gtRet)
    print('Retreival Acc ', predRet/gtRet )
    print('-'*10)
    print('Total Evaluated Res qa pairs ', gtRes)
    print('Reason Acc ', predRes/gtRes )
    
    print('-'*30)
    print('Total Evaluated Line qa pairs ', gtLine)
    print('Line Acc ', predLine/gtLine )
    
    print('-'*10)
    print('Total Evaluated VBar qa pairs ', gtVbar)
    print('Vbar Acc ', predVbar/gtVbar )
    
    print('-'*10)
    print('Total Evaluated Hbar qa pairs ', gtHbar)
    print('Hbar Acc ', predHbar/gtHbar )
    
    print('-'*10)
    print('Total Evaluated Scatter qa pairs ', gtScat)
    print('Sca Acc ', predScat/gtScat )
    
    print('-'*10)
    print('Total Evaluated VBox qa pairs ', gtVbox)
    print('box Acc ', predVbox/gtVbox )
    print('flat - ', predLine/gtLine, predVbar/gtVbar, predHbar/gtHbar, predScat/gtScat,  predVbox/gtVbox,  )
    print('-'*80)
    print('Sample 2')
    print('-'*80)
    s2 = list(np.load('./s2.npy'))
    
    gtstring = len([_ for _ in s2 if _ in score_obj['string_score']])
    predstring =  sum([score_obj['string_score'][_] for _ in s2 if _ in score_obj['string_score']])

    gtnum = len([_ for _ in s2 if _ in score_obj['num_score']])
    prednum = sum([score_obj['num_score'][_] for _ in s2 if _ in score_obj['num_score']])

    gtrank = len([_ for _ in s2 if _ in score_obj['rank_score']])
    predrank = sum([score_obj['rank_score'][_] for _ in s2 if _ in score_obj['rank_score']])

    gturank = len([_ for _ in s2 if _ in score_obj['urank_score']])
    predurank =  sum([score_obj['urank_score'][_] for _ in s2 if _ in score_obj['urank_score']])

    gtbi = len([_ for _ in s2 if _ in score_obj['bi_score']])
    predbi =  sum([score_obj['bi_score'][_] for _ in s2 if _ in score_obj['bi_score']])



    totalgt  = gtstring +  gtnum + gtrank + gturank + gtbi
    totalpred  = predstring + prednum + predrank + predurank + predbi
    print('Total Evaluated charts ', cix+1-count_pred)
    print('Total Evaluated qa pairs ', totalgt)
    print('Total Accuracy ', totalpred/totalgt)
    print('-'*10)
    print('Total Evaluated String qa pairs ',gtstring )
    print('String Acc ', predstring/gtstring )
    print('-'*10)
    print('Total Evaluated Numeric qa pairs ', gtnum)
    print('Numeric Acc ', prednum/gtnum )
    print('-'*10)
    print('Total Evaluated Ranked qa pairs ', gtrank)
    print('Rank Acc ', predrank/gtrank )
    print('-'*10)
    print('Total Evaluated Unranked qa pairs ', gturank)
    print('Urank Acc ', predurank/gturank )
    print('-'*10)
    print('Total Evaluated Binary qa pairs ', gtbi)
    print('Bi Acc ', predbi/gtbi )
    print('flat - ', totalpred/totalgt, predstring/gtstring, prednum/gtnum, predrank/gtrank, predurank/gturank, predbi/gtbi)

    print('-'*80)
    print('Sample 3')
    print('-'*80)
    s3 = list(np.load('./s3.npy'))
    
    gtstring = len([_ for _ in s3 if _ in score_obj['string_score']])
    predstring =  sum([score_obj['string_score'][_] for _ in s3 if _ in score_obj['string_score']])

    gtnum = len([_ for _ in s3 if _ in score_obj['num_score']])
    prednum = sum([score_obj['num_score'][_] for _ in s3 if _ in score_obj['num_score']])

    gtrank = len([_ for _ in s3 if _ in score_obj['rank_score']])
    predrank = sum([score_obj['rank_score'][_] for _ in s3 if _ in score_obj['rank_score']])

    gturank = len([_ for _ in s3 if _ in score_obj['urank_score']])
    predurank =  sum([score_obj['urank_score'][_] for _ in s3 if _ in score_obj['urank_score']])

    gtbi = len([_ for _ in s3 if _ in score_obj['bi_score']])
    predbi =  sum([score_obj['bi_score'][_] for _ in s3 if _ in score_obj['bi_score']])

    totalgt  = gtstring +  gtnum + gtrank + gturank + gtbi
    totalpred  = predstring + prednum + predrank + predurank + predbi
    print('Total Evaluated charts ', cix+1-count_pred)
    print('Total Evaluated qa pairs ', totalgt)
    print('Total Accuracy ', totalpred/totalgt)
    print('-'*10)
    print('Total Evaluated String qa pairs ',gtstring )
    print('String Acc ', predstring/gtstring )
    print('-'*10)
    print('Total Evaluated Numeric qa pairs ', gtnum)
    print('Numeric Acc ', prednum/gtnum )
    print('-'*10)
    print('Total Evaluated Ranked qa pairs ', gtrank)
    print('Rank Acc ', predrank/gtrank )
    print('-'*10)
    print('Total Evaluated Unranked qa pairs ', gturank)
    print('Urank Acc ', predurank/gturank )
    print('-'*10)
    print('Total Evaluated Binary qa pairs ', gtbi)
    print('Bi Acc ', predbi/gtbi )
    print('flat - ', totalpred/totalgt, predstring/gtstring, prednum/gtnum, predrank/gtrank, predurank/gturank, predbi/gtbi)



    

    print('-'*80)
    print('Sample 4')
    print('-'*80)
    s3 = list(np.load('./s4.npy'))
    
    gtstring = len([_ for _ in s3 if _ in score_obj['string_score']])
    predstring =  sum([score_obj['string_score'][_] for _ in s3 if _ in score_obj['string_score']])

    gtnum = len([_ for _ in s3 if _ in score_obj['num_score']])
    prednum = sum([score_obj['num_score'][_] for _ in s3 if _ in score_obj['num_score']])

    gtrank = len([_ for _ in s3 if _ in score_obj['rank_score']])
    predrank = sum([score_obj['rank_score'][_] for _ in s3 if _ in score_obj['rank_score']])

    gturank = len([_ for _ in s3 if _ in score_obj['urank_score']])
    predurank =  sum([score_obj['urank_score'][_] for _ in s3 if _ in score_obj['urank_score']])

    gtbi = len([_ for _ in s3 if _ in score_obj['bi_score']])
    predbi =  sum([score_obj['bi_score'][_] for _ in s3 if _ in score_obj['bi_score']])

    totalgt  = gtstring +  gtnum + gtrank + gturank + gtbi
    totalpred  = predstring + prednum + predrank + predurank + predbi
    print('Total Evaluated charts ', cix+1-count_pred)
    print('Total Evaluated qa pairs ', totalgt)
    print('Total Accuracy ', totalpred/totalgt)
    print('-'*10)
    print('Total Evaluated String qa pairs ',gtstring )
    print('String Acc ', predstring/gtstring )
    print('-'*10)
    print('Total Evaluated Numeric qa pairs ', gtnum)
    print('Numeric Acc ', prednum/gtnum )
    print('-'*10)
    print('Total Evaluated Ranked qa pairs ', gtrank)
    print('Rank Acc ', predrank/gtrank )
    print('-'*10)
    print('Total Evaluated Unranked qa pairs ', gturank)
    print('Urank Acc ', predurank/gturank )
    print('-'*10)
    print('Total Evaluated Binary qa pairs ', gtbi)
    print('Bi Acc ', predbi/gtbi )
    print('flat - ', totalpred/totalgt, predstring/gtstring, prednum/gtnum, predrank/gtrank, predurank/gturank, predbi/gtbi)



    print('-'*80)
    print('Sample 5')
    print('-'*80)
    s3 = list(np.load('./s5.npy'))
    
    gtstring = len([_ for _ in s3 if _ in score_obj['string_score']])
    predstring =  sum([score_obj['string_score'][_] for _ in s3 if _ in score_obj['string_score']])

    gtnum = len([_ for _ in s3 if _ in score_obj['num_score']])
    prednum = sum([score_obj['num_score'][_] for _ in s3 if _ in score_obj['num_score']])

    gtrank = len([_ for _ in s3 if _ in score_obj['rank_score']])
    predrank = sum([score_obj['rank_score'][_] for _ in s3 if _ in score_obj['rank_score']])

    gturank = len([_ for _ in s3 if _ in score_obj['urank_score']])
    predurank =  sum([score_obj['urank_score'][_] for _ in s3 if _ in score_obj['urank_score']])

    gtbi = len([_ for _ in s3 if _ in score_obj['bi_score']])
    predbi =  sum([score_obj['bi_score'][_] for _ in s3 if _ in score_obj['bi_score']])

    totalgt  = gtstring +  gtnum + gtrank + gturank + gtbi
    totalpred  = predstring + prednum + predrank + predurank + predbi
    print('Total Evaluated charts ', cix+1-count_pred)
    print('Total Evaluated qa pairs ', totalgt)
    print('Total Accuracy ', totalpred/totalgt)
    print('-'*10)
    print('Total Evaluated String qa pairs ',gtstring )
    print('String Acc ', predstring/gtstring )
    print('-'*10)
    print('Total Evaluated Numeric qa pairs ', gtnum)
    print('Numeric Acc ', prednum/gtnum )
    print('-'*10)
    print('Total Evaluated Ranked qa pairs ', gtrank)
    print('Rank Acc ', predrank/gtrank )
    print('-'*10)
    print('Total Evaluated Unranked qa pairs ', gturank)
    print('Urank Acc ', predurank/gturank )
    print('-'*10)
    print('Total Evaluated Binary qa pairs ', gtbi)
    print('Bi Acc ', predbi/gtbi )
    print('flat - ', totalpred/totalgt, predstring/gtstring, prednum/gtnum, predrank/gtrank, predurank/gturank, predbi/gtbi)


    
    # print('-'*80)
    # print('Sample 3')
    # s3 = np.load('/home/sahmed9/reps/cqa/s3.npy')
    
    # print('-'*80)
    # print('Sample 4')
    # s4 = np.load('/home/sahmed9/reps/cqa/s4.npy')
    
    # print('-'*80)
    # print('Sample 5')
    # s5 = np.load('/home/sahmed9/reps/cqa/s5.npy')
    
    # # print('-'*80)
    # print('-'*10)

    t = str(int(time.time()))
    if sd is not None : 
        sd = os.path.join(sd, 'score_matrix_'+t+'.npy')
        print('Saving score Matrix to', sd)
        with open(sd, 'wb') as f : 
            np.save(f, score_obj)
     
 


if __name__ == "__main__": 

    pred_dir = None
    # pred_dir = '/home/sahmed9/reps/cqa/distilbert_base_cased_distilled_squad'
    # pred_dir = '/home/sahmed9/reps/cqa/deepset/roberta-base-squad2'
    pred_dir = '/home/csgrad/sahmed9/reps/RealCQA/output_matcha_chartqa'


    # pr_js = json.load(open('/home/sahmed9/reps/cqa/op_bhavin_with_id_crct_baseline/CRCT_Baseline_Predictions.json'))
    # pr_js = json.load(open('/home/bhavinja/ChartInfo/Chart_VQA_ICDAR/CRCT/CQA-CRCT/CRCT/data/PMC/CRCT_Finetuned_Predictions.json'))
    # pr_js = json.load(open('/home/bhavinja/ChartInfo/Chart_VQA_ICDAR/CRCT/CQA-CRCT/CRCT/data/PMC/CRCT_Finetuned_Predictions_19th.json'))
    # pr_js = json.load(open('/home/sahmed9/reps/cqa/op_shubham/ptvlt5_chartqa_formatted.json'))
    # pr_js = json.load(open('/home/sahmed9/reps/cqa/op_shubham/ftvlt5_chartqa_formatted.json'))
    # pr_js = json.load(open('predictionsorigvlt5_formatted.json'))
    # pr_js = json.load(open('/home/sahmed9/reps/cqa/predictionsftvlt5_formatted.json'))
    # pr_js = json.load(open('C:/Users/spandey8/Downloads/ptvlt5_yes_bias.json'))
    # pr_js = json.load(open('C:/Users/spandey8/Downloads/ptvlt5_yes_bias.json'))
    
    # pr_obj = {}
    # pr_obj = None

    # for pr in pr_js :          
    #     pr_obj.update({pr['qa_id'] : pr})
    # print('Total Predicted QA', len(pr_obj))
    # pr_obj = None
    # gt_dir = 'C:/Users/spandey8/Desktop/llm_multi_modal_data/pmctest22/'
    gt_dir = "/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/cqa_22_with_id/combined"
    # sd = 'C:/Users/spandey8/Desktop/'
    sd = None
    # run_eval(pred_dir, gt_dir, sd, pr_obj)
    run_eval(pred_dir, gt_dir, sd)




"""


ix_map_ctpe = {}

for t in td : 
    props = [] 
    ix_map.update({t['qa_id'] : props})


s2, s3, s4, s5 = [], [], [], []

for s in os.listdir('sample2') :
    js = json.load(open(os.path.join('sample2', s)))
    for j in js : 
        s2.append(j['qa_id']) 




for s in os.listdir('sample3') :
    js = json.load(open(os.path.join('sample3', s)))
    for j in js : 
        s3.append(j['qa_id']) 




for s in os.listdir('sample4') :
    js = json.load(open(os.path.join('sample4', s)))
    for j in js : 
        s4.append(j['qa_id']) 



for s in os.listdir('sample5') :
    js = json.load(open(os.path.join('sample5', s)))
    for j in js : 
        s5.append(j['qa_id']) 




tid = {}
for s in os.listdir('pmctest22') :
    js = json.load(open(os.path.join('pmctest22', s)))
    for j in js : 
        if j['taxonomy id'] in tid : 
            tid[j['taxonomy id']].append(j['qa_id'])
        else :
            tid.update({j['taxonomy id']:[j['qa_id']]})




fd = '/home/sahmed9/Documents/icdar23/chart_data_extraction/ICPR2022_CHARTINFO_UB_UNITEC_PMC_TEST_v2.1/flat/'

ctid = {}
for s in os.listdir('pmctest22') :
    js = json.load(open(os.path.join('pmctest22', s)))
    jobj = json.load(open(os.path.join(fd, s))) 
    for j in js :
        ctpy_ = jobj['task1']['output']['chart_type']
        if ctpy_ in ctid:
            ctid[ctpy_].append(j['qa_id'])
        else : 
            ctid.update({ctpy_ :[j['qa_id']]})




##########################################


leg = [ 'Vertical Bar', 'Line', 'Scatter','Horizonal Bar',  'Vertical Box']
xti = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6']
v = [3745,562,562,496,502,427]
l = [ 3399,623,623,457,453,398]
s = [1247, 298, 298, 141, 140, 103]
h = [ 634, 209,209,146,148,113]
vb = [775, 179, 179, 110, 109, 92]

plt.plot(h)
plt.plot(v)
plt.plot(l)
plt.plot(s)
plt.plot(vb)


plt.bar(xti,v)
plt.bar(xti,l)
plt.bar(xti,s)
plt.bar(xti,h)
plt.bar(xti,vb)


plt.bar(xti[1:],v[1:])
plt.bar(xti[1:],l[1:])
plt.bar(xti[1:],s[1:])
plt.bar(xti[1:],h[1:])
plt.bar(xti[1:],vb[1:])



plt.plot(v)
plt.plot(l)
plt.plot(s)
plt.plot(vb)

fig, ax = plt.subplots()
x = np.arange(0, 12, 2)
ax.bar(x-0.8, v, width)
ax.bar(x-0.4, l, width)
ax.bar(x, s, width)
ax.bar(x+0.4, h, width)
ax.bar(x+0.8, vb, width)
ax.legend(leg)
ax.set_xticks(x , xti)
ax.set_ylabel('Charts Count Available for QA')
ax.set_xlabel('Chart Challenge Tasks')

"""
