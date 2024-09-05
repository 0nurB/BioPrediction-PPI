from numba import jit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
import joblib
import networkx as nx
import dgl
import torch
import statistics
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score, f1_score,
    auc, accuracy_score, balanced_accuracy_score, roc_auc_score
)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
#import lightgbm as lgb
import xgboost as xgb
from dgl import DGLGraph
from dgl.nn import SAGEConv
from node2vec import Node2Vec
from Bio import SeqIO
import argparse
import subprocess
import sys
import os.path
from subprocess import Popen
from multiprocessing import Manager
import polars as pl
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score, make_scorer
from hyperopt import STATUS_OK, STATUS_FAIL
from sklearn.exceptions import FitFailedWarning
import shap
from interpretability_report import Report, REPORT_MAIN_TITLE_BINARY, REPORT_SHAP_PREAMBLE_BINARY, REPORT_SHAP_BAR_BINARY, \
    REPORT_SHAP_BEESWARM_BINARY, REPORT_SHAP_WATERFALL_BINARY

from utility_graphs import REPORT_USABILITY_TITLE_BINARY, REPORT_USABILITY_DESCRIPITION, \
    REPORT_1, REPORT_2, REPORT_3

from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from numpy.random import default_rng

from bio_extrac import extrac_math_features, graph_ext, make_graph, make_graph_all
from utility_graphs import final_predictions, precision_graph, coverage_graph, make_fig_graph
from shap_graphs import type_model, shap_waterf, shap_bar, shap_beeswarm
from make_model import metrics, better_model, Score_table, objective_rf, tuning_rf_bayesian, objective_cb, tuning_catboost_bayesian, objective_gb, tuning_xgb_bayesian


warnings.filterwarnings("ignore", message="X has feature names, but DecisionTreeClassifier was fitted without feature names")
warnings.filterwarnings("ignore", message="X has feature names, but RandomForestClassifier was fitted without feature names")
warnings.filterwarnings('ignore')

def concat_data_models(datas, output, name, candidates_bol = False):
    """
    Concatenate data from multiple models and save the final training dataset.

    Parameters:
        datas (list): List of paths to CSV files containing data from multiple models.
        output (str): Directory where the final training data will be saved.
        name (str): Name or identifier for the concatenated dataset.

    Returns:
        final_X_train (DataFrame): Final training data features.
        final_y_train (DataFrame): Final training data labels.
    """
    # Read the first dataset file
    merged_train = pd.read_csv(datas[0], sep=',')
    merged_trainB = pd.read_csv(datas[1], sep=',')
    
    # Merge the first two datasets on the 'nameseq' column and drop the redundant 'Label_x'
    merged_train = merged_train.merge(merged_trainB, on='nameseq')
    if 'Label_x' in merged_train.columns:
        merged_train = merged_train.drop(columns=['Label_x'], axis=1)

    # Iterate through the remaining dataset files and merge them, dropping the 'Label' column each time
    for k in range(2, len(datas)):
        merged_trainB = pd.read_csv(datas[k], sep=',')
        if 'Label' in merged_trainB.columns:
            merged_trainB = merged_trainB.drop(columns=['Label'], axis=1)
        merged_train = merged_train.merge(merged_trainB, on='nameseq')
        merged_train = merged_train.drop_duplicates() 

    # Extract the 'nameseq' and 'Label_y' columns as final training data
    merged_nameseq = merged_train[['nameseq']].copy()
    if candidates_bol:
        final_y_train = pd.DataFrame()
        final_X_train = merged_train.drop(columns=['nameseq'])
        if 'Label_y' in merged_train.columns:
            final_X_train = final_X_train.drop(columns=['Label_y'])
    else:
        final_y_train = merged_train[['Label_y']].copy()
        final_X_train = merged_train.drop(columns=['nameseq', 'Label_y'])
        
    final_X_train = final_X_train.drop_duplicates()    
    final_y_train = final_y_train.drop_duplicates() 
    merged_nameseq= merged_nameseq.drop_duplicates()    
        
    final_X_train.to_csv(output+'/final_X_'+name+'.csv', index=False) 
    final_y_train.to_csv(output+'/final_y_'+name+'.csv', index=False) 
    merged_nameseq.to_csv(output+'/final_nameseq_'+name+'.csv', index=False)
    
    return final_X_train, final_y_train, merged_nameseq



def make_dataset(test_edges, carac):
    """
    Create a dataset for testing by merging test_edges with carac (characteristics) data.

    Parameters:
        test_edges (DataFrame): DataFrame containing test edges, including source and target nodes.
        carac (DataFrame): DataFrame containing characteristics data, indexed by node identifiers.

    Returns:
        X_test (DataFrame): Features of the test dataset.
        y_test (Series): Labels of the test dataset.
        test_data (DataFrame): Merged dataset for testing.
    """
    columns = test_edges.columns

    # Ensure that the index of carac is of string type
    carac.index = carac.index.astype(str)
    #print(carac)
    # Substitua 'nome_coluna_carac' pelos nomes reais da primeira coluna de carac
    carac[carac.columns[0]] = carac[carac.columns[0]].astype(str)

    # Substitua 'nome_coluna_train_edges1' e 'nome_coluna_train_edges2' pelos nomes reais das duas primeiras colunas de train_edges
    test_edges[test_edges.columns[:2]] = test_edges[test_edges.columns[:2]].astype(str)

    # Merge test_edges with carac based on source and target nodes
    #test_data = test_edges.merge(carac, left_on=columns[0], right_index=True).merge(carac, left_on=columns[1], right_index=True)
    test_data = (
        test_edges
        .merge(carac, left_on=columns[0], right_index=True, suffixes=('_A', '_B'))
        .merge(carac, left_on=columns[1], right_index=True, suffixes=('_A', '_B'))
    )
    # Remove duplicate rows from the merged dataset
    test_data = test_data.drop_duplicates()
    # Separate features (X_test) and labels (y_test)
    X_test = test_data.drop(columns, axis=1)
    X_test = X_test.astype(float)
    #print(X_test)
    #X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
    #print(test_data)
    y_test = test_data[columns[2]]
    y_test = y_test.astype(float)

    return X_test, y_test, test_data






# python reuse.py -trained_model_path yeast_new -input_interactions_candidates data_yeast/interaction.csv
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-trained_model_path', '--trained_model_path', help='...')
    parser.add_argument('-input_interactions_candidates', '--input_interactions_candidates', help='path of table with two columns (firts protein and secund RNA) of your candidates interactions pairs e.g., candidates.txt', default='')
    parser.add_argument('-topological_features', '--topological_features', default='yes', help='topology features were used to characterize the sequences, e.g., yes or no, default=yes)')
    parser.add_argument('-fold_prediction', '--fold_prediction', default=1, help='Choose the fold that will be used to predict the candidates, from 1 to 5, default=1')
    
     
    args = parser.parse_args()
    trained_model_path = args.trained_model_path
    foutput = trained_model_path
    candidates = args.input_interactions_candidates
    candidates = pd.read_csv(candidates, sep=",")
    topological_features = args.topological_features
    p = int(args.fold_prediction)
        
    if topological_features == 'no':
        extrac_topo_features = False
    elif topological_features == 'yes':
        extrac_topo_features = True
    else:
        print('Error: topological features difference between yes or no, will be considered no')
        extrac_topo_features = False
    
    if extrac_topo_features == True:
        names_topo = ['topology_P', 'topology_N']
        datasets_topo=[]
        for p1 in range(5):            
            datasets_topo.append([foutput+'/folds_and_topology_feats/fold'+str(p1+1)+'/feat_'+names_topo[0]+'.csv',foutput+'/folds_and_topology_feats/fold'+str(p1+1)+'/feat_'+names_topo[1]+'.csv'])
    
    
    datasets_extr = []
    names_math = []    
    names_math_feat=['merged_data', 'kGap_di', 'Shannon', 'Tsallis_23', 'Tsallis_30', 'Tsallis_40','AAC', 'DPC']
    #names_math_feat=['merged_data', 'Shannon', 'Tsallis_23', 'Tsallis_30', 'Tsallis_40','AAC', 'DPC']                     
    features_amino = [1,2,3,4,5,6,7,8]
     
    for i in range(len(names_math_feat)):
        names_math.append(names_math_feat[i])
        datasets_extr.append(foutput+'/extructural_feats/'+names_math_feat[i]+'.csv')
        
    cands = []
    print('Fold number', p, 'was used')
    if extrac_topo_features == True:
        for i in range(2):
            datasets = datasets_topo[p-1][i] 
            carac = pd.read_csv(datasets, sep=",")
            index = 'Node'
            carac.set_index(index, inplace=True)
            if 'label' in carac.columns:
                carac = carac.drop(columns=['label'])
            if 'Label' in carac.columns:
                carac = carac.drop(columns=['Label'])

            clf_path = trained_model_path+'/'+'folds_and_topology_feats/fold'+str(p)+'/partial_models'+'/model_'+names_topo[i]+'.sav'
            clf = joblib.load(clf_path)

            columns = candidates.columns
            candidates_names = candidates.copy()
            candidates[candidates.columns[:2]] = candidates[candidates.columns[:2]].astype(str)
            carac.index = carac.index.astype(str)
            candidates_data = candidates.merge(carac, left_on=columns[0], right_index=True).merge(carac, left_on=columns[1], right_index=True)
            candidates_names = candidates_data.iloc[:, :3]

            candidates_data = candidates_data.drop(columns=['ProteinA', 'ProteinB'])
            if 'Label' in candidates_data.columns:
                candidates_data = candidates_data.drop(columns=['Label'])
            candidates_data = candidates_data.astype(float)
            score = clf.predict_proba(candidates_data)[:, 1]
            predictions=[]
            Score_table(candidates_names, predictions, score, 'Score_'+names_topo[i] , trained_model_path+'/'+'folds_and_topology_feats/fold'+str(p)+'/partial_models'+'/data_candidates_'+names_topo[i]+'.csv')

            cands.append(trained_model_path+'/'+'folds_and_topology_feats/fold'+str(p)+'/partial_models'
                         +'/data_candidates_'+names_topo[i]+'.csv')
    
#################################################    
  
    
    for j in range(1,len(names_math_feat)):
        datasets = datasets_extr[j] 
        carac = pd.read_csv(datasets, sep=",")
        index = 'nameseq'
        carac.set_index(index, inplace=True)
        if 'label' in carac.columns:
            carac = carac.drop(columns=['label'])
        if 'Label' in carac.columns:
            carac = carac.drop(columns=['Label'])

        clf_path = trained_model_path+'/'+'folds_and_topology_feats/fold'+str(p)+'/partial_models'+'/model_'+names_math_feat[j]+'.sav'
        clf = joblib.load(clf_path)

        columns = candidates.columns

        candidates_names = candidates.copy()
        candidates[candidates.columns[:2]] = candidates[candidates.columns[:2]].astype(str)
        carac.index = carac.index.astype(str)
        candidates_data = candidates.merge(carac, left_on=columns[0], right_index=True).merge(carac, left_on=columns[1], right_index=True)
        candidates_names = candidates_data.iloc[:, :3]
        candidates_data = candidates_data.drop(columns=['ProteinA', 'ProteinB'])
        if 'Label' in candidates_data.columns:
            candidates_data = candidates_data.drop(columns=['Label'])
        candidates_data = candidates_data.astype(float)
        score = clf.predict_proba(candidates_data)[:, 1]
        predictions=[]
        Score_table(candidates_names, predictions, score, 'Score_'+names_math_feat[j], trained_model_path+'/'+'folds_and_topology_feats/fold'+str(p)+'/partial_models'+'/data_candidates_'+names_math_feat[j]+'.csv')

        cands.append(trained_model_path+'/'+'folds_and_topology_feats/fold'+str(p)+'/partial_models'
                     +'/data_candidates_'+names_math[j]+'.csv')

############################################        

    concat_output = trained_model_path+'/'+'folds_and_topology_feats/fold'+str(p)
    final_X_cand, trash, nameseq_cand = concat_data_models(cands, concat_output, 'cands', True)

    clf_path =  trained_model_path+'/'+'folds_and_topology_feats/fold'+str(p)+'/model_final.sav'
    clf = joblib.load(clf_path)
    merged_df = nameseq_cand[nameseq_cand.index.isin(final_X_cand.index)]

    predictions = clf.predict(final_X_cand)
    score = clf.predict_proba(final_X_cand)[:, 1]
    candidates['name_pair'] = candidates.iloc[:, 0] + '_' + candidates.iloc[:, 1]
    score_name = 'Candidate_score'
    output_table = trained_model_path+'/'+'folds_and_topology_feats/fold'+str(p)+'/candidates_prediction.csv'
    model_result = merged_df.copy()
    model_result[score_name] = score
    
    temp_df = candidates[['name_pair', 'Label']]
    model_result = model_result.merge(temp_df, left_on='nameseq', right_on='name_pair', how='left')
    model_result.drop('name_pair', axis=1, inplace=True)
    model_result.to_csv(output_table, index=False)
    print('Predicted and saved interactor candidates in', output_table)





