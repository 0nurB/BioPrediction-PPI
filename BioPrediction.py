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

scaler = StandardScaler()
warnings.filterwarnings('ignore')
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=40)


def fitting(test_data, score_name, clf_fit, X_test, y_test, output_table, output_metrics=None):
    """
    Fit a classifier, generate predictions and scores, and create a score table. Optionally, calculate metrics.

    Parameters:
        test_data (DataFrame): DataFrame containing test data, including sequence names.
        score_name (str): Name of the score column.
        clf_fit (Classifier): A trained classifier.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.
        output_table (str): File path where the score table will be saved.
        output_metrics (str, optional): File path where metrics will be saved. Default is None.

    Returns:
        model (DataFrame): The score table DataFrame.
    """
    # Generate predictions using the trained classifier
    predictions = clf_fit.predict(X_test)
    preds_proba = clf_fit.predict_proba(X_test.values)[:, 1]
    # Calculate and save metrics if output_metrics is provided
    if output_metrics is not None:
        metrics(predictions, y_test, preds_proba, output_metrics)

    # Generate probability scores for the positive class
    scores = clf_fit.predict_proba(X_test)[:, 1]

    # Create a score table with sequence names, predicted labels, and scores
    model = Score_table(test_data, predictions, scores, score_name, output_table)

    return model


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

def make_trains(train, test, final_test, carac):
    """
    Create training and testing datasets by merging edge data with characteristics data.

    Parameters:
        train (str): Path to the training data file (edges).
        test (str): Path to the testing data file (edges).
        final_test (str): Path to the final testing data file (edges).
        carac (DataFrame): DataFrame containing characteristics data, indexed by node identifiers.

    Returns:
        X_train (DataFrame): Features of the training dataset.
        y_train (Series): Labels of the training dataset.
        X_test (DataFrame): Features of the testing dataset.
        y_test (Series): Labels of the testing dataset.
        test_data (DataFrame): Merged dataset for testing.
        X_final (DataFrame): Features of the final testing dataset.
        y_final (Series): Labels of the final testing dataset.
        test_data_final (DataFrame): Merged dataset for final testing.
    """
    # Read training, testing, and final testing data from CSV files
    train_edges = pd.read_csv(train, sep=',')
    test_edges = pd.read_csv(test, sep=',')
    final_test = pd.read_csv(final_test, sep=',')

    # Create datasets for training, testing, and final testing
    X_train, y_train, test_data = make_dataset(train_edges, carac)
    X_test, y_test, test_data = make_dataset(test_edges, carac)
    X_final, y_final, test_data_final = make_dataset(final_test, carac)

    # Drop duplicate rows in the characteristics data
    #print(carac, 'carac')
    #carac = carac.drop_duplicates()
    #print(carac, 'carac_dp')
    
    return X_train, y_train, X_test, y_test, test_data, X_final, y_final, test_data_final


def partial_models(index, datasets, names, train_edges, test_edges, final_test, partial_folds):
    """
    Create and save partial models and associated data for a specific dataset.

    Parameters:
        index (str): Name of the index column in datasets.
        datasets (str): Path to the dataset file containing characteristics data.
        names (str): Name or identifier for the dataset.
        train_edges (str): Path to the training data file (edges).
        test_edges (str): Path to the testing data file (edges).
        final_test (str): Path to the final testing data file (edges).
        partial_folds (str): Directory where partial models and data will be saved.

    Returns:
        None
    """
    # Read characteristics data from CSV file and set index
    carac = pd.read_csv(datasets, sep=",")
    carac.set_index(index, inplace=True)

    # Check if 'label' column exists in carac and drop it if present
    if 'label' in carac.columns:
        carac = carac.drop(columns='label')

    # Create training, testing, and final testing datasets
    X_train, y_train, X_test, y_test, test_data, X_final, y_final, test_data_final = \
        make_trains(train_edges, test_edges, final_test, carac)

    # Train a classifier, save the model, and generate score tables and metrics
    clf = better_model(X_train, y_train, X_test, y_test, partial_folds+'/model_'+names+'.sav')      
    
    fitting(test_data, 'Score_'+names, clf, X_test, y_test, partial_folds+'/data_train_'+names+'.csv', partial_folds+'/metrics_'+names+'.csv')        
    
    fitting(test_data_final, 'Score_'+names, clf, X_final, y_final,
            partial_folds+'/data_test_'+names+'.csv')
    
    if isinstance(candidates, pd.DataFrame):       
        columns = candidates.columns
        candidates_names = candidates.copy()

        if 'label' in carac.columns:
            df = carac.drop(columns=['label'])
        if 'Label' in carac.columns:
            df = carac.drop(columns=['Label'])     
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
        Score_table(candidates_names, predictions, score, 'Score_'+names, partial_folds+'/data_candidates_'+names+'.csv')
        
    y_train = pd.DataFrame(y_train)
    y_train.rename(columns={y_train.columns[0]: 'Label_y'}, inplace=True)
    generated_plt = interp_shap(clf, X_train, y_train, partial_folds, names) 
    #build_interpretability_report(generated_plt=generated_plt, directory=partial_folds)



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
        
    duplicate_indices = final_X_train.index[final_X_train.index.duplicated()]


    # Remover as linhas com esses índices duplicados de todos os DataFrames
    final_X_train = final_X_train.drop(index=duplicate_indices)
    final_y_train = final_y_train.drop(index=duplicate_indices)
    merged_nameseq = merged_nameseq.drop(index=duplicate_indices)
        
    final_X_train.to_csv(output+'/final_X_'+name+'.csv', index=False) 
    final_y_train.to_csv(output+'/final_y_'+name+'.csv', index=False) 
    merged_nameseq.to_csv(output+'/final_nameseq_'+name+'.csv', index=False)
    
    return final_X_train, final_y_train, merged_nameseq

def interp_shap(model, X_test, X_label, output, path='explanations', name=''):
    """
    Generate various types of SHAP interpretation graphs for a given model and dataset.
    
    Parameters:
    - model: The machine learning model to be explained.
    - X_test: The test data for which SHAP values will be calculated.
    - X_label: The labels or target values associated with the test data.
    - output: The output directory where the explanation graphs will be saved.
    - path: The subdirectory within 'output' where the explanation graphs will be saved (default is 'explanations').

    Returns:
    - generated_plt: A dictionary containing the file paths of the generated explanation graphs.
    """
    path = os.path.join(output, path)
    generated_plt = {}
    
    # Create a SHAP explainer for the model with tree-based feature perturbation.
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    
    # Calculate SHAP values and potentially transform labels based on model type.
    shap_values, X_label = type_model(explainer, model, X_test, X_label)
    
    if not os.path.exists(path):
        #print(f"Creating explanations directory: {path}...")
        os.mkdir(path)
    #else:
        #print(f"Directory {path} already exists. Will proceed using it...")
        
    if name != '':
        fig_name1 = 'bar_graph_'+name
        fig_name2='beeswarm_graph_'+name
    else:
        fig_name1 = 'bar_graph'
        fig_name2='beeswarm_graph'
    generated_plt['bar_graph'] = [shap_bar(shap_values, path, fig_name1)]
    generated_plt['beeswarm_graph'] = [shap_beeswarm(shap_values, path, fig_name2)]
    generated_plt['waterfall_graph'] = shap_waterf(explainer, model, X_test, X_label, path, name)
    return generated_plt


   
def build_interpretability_report(generated_plt=[], report_name="interpretability.pdf", directory="."):
    """
    Build an interpretability report by combining generated SHAP interpretation graphs into a PDF report.
    
    Parameters:
    - generated_plt: A dictionary containing file paths of generated explanation graphs.
    - report_name: The name of the PDF report to be generated (default is "interpretability.pdf").
    - directory: The directory where the PDF report will be saved (default is the current directory).
    """
    report = Report(report_name, directory=directory)
    root_dir = os.path.abspath(os.path.join(__file__, os.pardir))

    report.insert_doc_header(REPORT_MAIN_TITLE_BINARY, logo_fig=os.path.join(root_dir, "img/BioAutoML.png"))
    report.insert_text_on_doc(REPORT_SHAP_PREAMBLE_BINARY, font_size=14)

    report.insert_figure_on_doc(generated_plt['bar_graph'])
    report.insert_text_on_doc(REPORT_SHAP_BAR_BINARY, font_size=14)

    report.insert_figure_on_doc(generated_plt['beeswarm_graph'])
    report.insert_text_on_doc(REPORT_SHAP_BEESWARM_BINARY, font_size=12)

    report.insert_figure_on_doc(generated_plt['waterfall_graph'])
    report.insert_text_on_doc(REPORT_SHAP_WATERFALL_BINARY, font_size=12)

    report.build()
    
def utility(output_data, output):
    generated_plt = {}
    output = output + '/utility'
    make_fold(output)
    
    generated_plt['precision'] = precision_graph(output_data, output)
    generated_plt['coverage'] = coverage_graph(output_data, output)
    make_fig_graph(output_data, output)
    return generated_plt

def build_usability_report(generated_plt=[], report_name="usability.pdf", directory="."):
    """
    Build an interpretability report by combining generated SHAP interpretation graphs into a PDF report.
    
    Parameters:
    - generated_plt: A dictionary containing file paths of generated explanation graphs.
    - report_name: The name of the PDF report to be generated (default is "interpretability.pdf").
    - directory: The directory where the PDF report will be saved (default is the current directory).
    """
    report = Report(report_name, directory=directory)
    root_dir = os.path.abspath(os.path.join(__file__, os.pardir))

    report.insert_doc_header(REPORT_USABILITY_TITLE_BINARY, logo_fig=os.path.join(root_dir, "img/BioAutoML.png"))
    report.insert_text_on_doc(REPORT_USABILITY_DESCRIPITION, font_size=14)

    #report.insert_figure_on_doc(generated_plt['precision'])
    report.insert_text_on_doc(REPORT_1, font_size=14)
    
    #report.insert_figure_on_doc(generated_plt['coverage'])
    report.insert_text_on_doc(REPORT_1, font_size=14)
    report.build()

def real_part(complex_num):
    return complex_num.real

def extrac_features_topology(train_edges, edges, output):
    """
    Extract various graph topology features and save then to CSV files.

    Parameters:
        train_edges (pd.DataFrame): DataFrame containing training edge data.
        edges (pd.DataFrame): DataFrame containing edge information.
        output (str): Output directory for saving feature files.
    """

    print(f'Extracting topological features...')
    
    feat_output = output+'/feat_topology_P.csv'
    
    # Create a graph from training edges (considering positive edges only)
    G_trP = make_graph(train_edges, edges)    
    data_nodes_trainP = graph_ext(G_trP)    
    graph_feat_trainP = pd.DataFrame(data_nodes_trainP)
    graph_feat_trainP.set_index('Node', inplace=True)
    
    # Extract real parts of complex numbers (if any) in the DataFrame
    graph_feat_trainP = graph_feat_trainP.applymap(real_part) 
    
    # Save the graph features to a CSV file
    feat_output = output+'/feat_topology_P.csv'
    graph_feat_trainP.to_csv(feat_output, index=True)

    # Create a graph from training edges (considering all edges, including negative)
    G_trN = make_graph_all(train_edges, edges)
    data_nodes_trainN = graph_ext(G_trN)
    graph_feat_trainN = pd.DataFrame(data_nodes_trainN)
    graph_feat_trainN.set_index('Node', inplace=True)
    
    # Extract real parts of complex numbers (if any) in the DataFrame
    graph_feat_trainN = graph_feat_trainN.applymap(real_part)
    
    # Save the graph features to a CSV file
    feat_output = output+'/feat_topology_N.csv'
    graph_feat_trainN.to_csv(feat_output, index=True)
        

def check_path(paths, type_path='This path'):
    for subpath in paths:
        if os.path.exists(subpath):
            print(f'{type_path} - {subpath}: Found File')
        else:
            print(f'{type_path} - {subpath}: File not exists')
            sys.exit()            

def debug_path(path_input):
    path_output = []
    for i in range(len(path_input)):
        path_output.append(os.path.join(path_input[i]))
    return path_output

def make_fold(path_results):
    if not os.path.exists(path_results):
        os.mkdir(path_results)
        
def feat_eng(input_interactions_train, sequences_dictionary, n_cpu, foutput, candidates, topological_features):
    input_interactions_train = pd.read_csv(input_interactions_train, sep=',')
    
    global train_edges_output,  test_edges_output, final_edges_output, partial_folds, output_folds, output_folds_number, features_amino
    
    if topological_features == 'no':
        extrac_topo_features = False
    elif topological_features == 'yes':
        extrac_topo_features = True
    else:
        print('Error: topological features difference between yes or no, will be considered no')
        extrac_topo_features = False
    
    extrac_math_featuresB = False
    
    output_folds = foutput+'/folds_and_topology_feats'
    make_fold(output_folds)

    feat_path = foutput + '/extructural_feats'
    make_fold(feat_path)

    print('Make the folds')

    edges = input_interactions_train[input_interactions_train.columns]
    if isinstance(candidates, pd.DataFrame): 
        edges = pd.concat([input_interactions_train, candidates], ignore_index=True)
    
    train_edges_output = []
    test_edges_output = []
    final_edges_output = []
    output_folds_number = []
    partial_folds = []

    
    for fold, (train_index, test_index) in enumerate(kf.split(input_interactions_train)):
        train_edges = input_interactions_train.iloc[train_index]
        final_test = input_interactions_train.iloc[test_index]
        train_edges, test_edges = train_test_split(train_edges, test_size=1/8, random_state=42)
        output_folds_number.append(output_folds +'/fold'+str(fold+1))
        make_fold(output_folds_number[fold])

        partial_folds.append(output_folds_number[fold]+'/partial_models')
        make_fold(partial_folds[fold])

        train_edges_output.append(output_folds_number[fold]+'/edges_train.csv')
        train_edges.to_csv(train_edges_output[fold], index=False)
        
        test_edges_output.append(output_folds_number[fold]+'/edges_test.csv')
        test_edges.to_csv(test_edges_output[fold], index=False)

        final_edges_output.append(output_folds_number[fold]+'/edges_final_test.csv')
        final_test.to_csv(final_edges_output[fold], index=False)

        if extrac_topo_features:
            print('Topology features extraction fold'+str(fold+1))
            #if fold+1 == 10:
            extrac_features_topology(train_edges, edges, output_folds_number[fold])
        

        
    names_topo = ['topology_P', 'topology_N']
    datasets_topo=[]
    for p in range(num_folds):            
            datasets_topo.append([foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/feat_'+names_topo[0]+'.csv',
             foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/feat_'+names_topo[1]+'.csv'])

    features_amino = [1,2,3,4,5,6,7,8] #,9,10,11]
    if extrac_math_featuresB:  
        datasets_extr, names_math = extrac_math_features(features_amino, sequences_dictionary, feat_path)

    datasets_extr = []
    names_math = []
    names_math_feat=['merged_data', 'Shannon', 'Tsallis_23', 'Tsallis_30', 'Tsallis_40',
                      'kGap_di', 'AAC', 'DPC', 'feat_H1', 'feat_H2',
                   'feat_NCI', 'feat_P1', 'feat_P2', 'feat_SASA', 'feat_V', 'fq_grups'
                  , 'Mean_feat', 'merged_data' 'Mean_feat']

    for i in range(len(names_math_feat)):
        names_math.append(names_math_feat[i-1])
        datasets_extr.append(foutput+'/extructural_feats/'+names_math_feat[i]+'.csv')    
        

def fit_mod(input_interactions_train, sequences_dictionary, n_cpu, foutput, candidates, topological_features):
    parcial_models_cond = True
    final_model = True
    
    
    print('Starting the model training stage')
    
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
        for p in range(num_folds):            
                datasets_topo.append([foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/feat_'+names_topo[0]+'.csv',
                 foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/feat_'+names_topo[1]+'.csv'])
    
    datasets_extr = []
    names_math = []
    
    #names_math_feat=['merged_data', 'Shannon', 'Tsallis_23', 'Tsallis_30', 'Tsallis_40',
    #                  'AAC', 'DPC', 'feat_H1', 'feat_H2',
    #               'feat_NCI', 'feat_P1', 'feat_P2', 'feat_SASA', 'feat_V', 'fq_grups'
    #              ,'Mean_feat'] #, 'merged_data', 'Mean_feat'] #  'Global', 'Peptide', 'iFeature',

    names_math_feat=['merged_data', 'kGap_di', 'Shannon', 'Tsallis_23', 'Tsallis_30', 'Tsallis_40',
                      'AAC', 'DPC']
    features_amino = [1,2,3,4,5,6,7,8] #,9,10,11]

    for i in range(len(names_math_feat)):
        names_math.append(names_math_feat[i])
        datasets_extr.append(foutput+'/extructural_feats/'+names_math_feat[i]+'.csv') 
        
    if parcial_models_cond: 
        #print('Starting the model training stage')
        for p in range(num_folds): 
            print()
            trains = []
            tests = []
            cands = []

            if extrac_topo_features == True:
                for i in range(2):
                    partial_models('Node', datasets_topo[p][i], names_topo[i], train_edges_output[p], test_edges_output[p], final_edges_output[p], partial_folds[p])
                    trains.append(partial_folds[p]+'/data_train_'+names_topo[i]+'.csv')
                    tests.append(partial_folds[p]+'/data_test_'+names_topo[i]+'.csv')
                    if isinstance(candidates, pd.DataFrame): 
                        cands.append(partial_folds[p]+'/data_candidates_'+names_topo[i]+'.csv')
            
            for j in range(1,len(names_math_feat)):
                partial_models('nameseq', datasets_extr[j], names_math[j], train_edges_output[p], test_edges_output[p], final_edges_output[p], partial_folds[p])
                trains.append(partial_folds[p]+'/data_train_'+names_math[j]+'.csv')
                tests.append(partial_folds[p]+'/data_test_'+names_math[j]+'.csv')
                if isinstance(candidates, pd.DataFrame): 
                    cands.append(partial_folds[p]+'/data_candidates_'+names_math[j]+'.csv')


            concat_output = output_folds +'/fold'+str(p+1)
            final_X_train, final_y_train, nameseq = concat_data_models(trains, concat_output, 'train')
            final_X_test, final_y_test, nameseq = concat_data_models(tests, concat_output, 'test')
            if isinstance(candidates, pd.DataFrame): 
                final_X_cand, trash, nameseq_cand = concat_data_models(cands, concat_output, 'cands', True)

            global X_train, y_train
            X_train = final_X_train
            y_train = final_y_train

            clf = better_model(final_X_train, final_y_train, final_X_test, final_y_test, 
                               output_folds_number[p]+'/model_final.sav', tuning=False)


            generated_plt = interp_shap(clf, final_X_train, final_y_train, output_folds_number[p]) 
            build_interpretability_report(generated_plt=generated_plt, directory=output_folds_number[p])

            output_data = final_predictions(final_X_test, final_y_test, nameseq, clf, output_folds_number[p])
            generated_plt = utility(output_data, output_folds_number[p])
            build_usability_report(generated_plt, report_name="usability.pdf", directory=output_folds_number[p])

            predictions = clf.predict(final_X_test)
            preds_proba = clf.predict_proba(final_X_test)[:, 1]
            print('The metrics for fold', p+1, 'are available in', output_folds_number[p] + "/metrics_model_final.csv")
            metrics(predictions, final_y_test, preds_proba, output_folds_number[p]+'/metrics_model_final.csv')
            
            if isinstance(candidates, pd.DataFrame):
                predictions = clf.predict(final_X_cand)
                score = clf.predict_proba(final_X_cand)[:, 1]
                score_name = 'Candidate_score'
                output_table = output_folds_number[p]+'/candidates_prediction.csv'
                model_result = nameseq_cand.copy()
                model_result[score_name] = score
                print('The predictions of the candidate interactions are available in', output_table)
                model_result.to_csv(output_table, index=False)

    metrics_output = []
    for p in range(num_folds):
        metrics_output.append(foutput+'/folds_and_topology_feats/fold'+str(p+1)+'/metrics_model_final.csv')

    if final_model:
        print()
        print('Calculating the average of metrics for the five folds')
        metrics1 = pd.read_csv(metrics_output[0], sep=',')
        for i in range(1, len(metrics_output)):  # Start from 1 as you've already read the first file
            metrics2 = pd.read_csv(metrics_output[i], sep=',')
            metrics1 = pd.concat([metrics1, metrics2], ignore_index=True)

        new_dataframe = pd.DataFrame(columns=['Metric', 'Mean', 'Standard deviation'])
        metric_names = ['Precision_1', 'Recall_1', 'F1_1', 'Specificity_1', 'Precision_0', 
                        'Recall_0', 'F1_0', 'Accuracy', 'AUPR', 'Balanced_Accuracy', 'MCC']

        for metric_name in metric_names:
            metric_data = metrics1[metrics1['Metrics'] == metric_name]
            mean_value = metric_data['Values'].mean()
            std_value = metric_data['Values'].std()
            new_line = {'Metric': metric_name, 'Mean': mean_value, 'Standard deviation': std_value}    
            new_dataframe = new_dataframe._append(new_line, ignore_index=True)

        new_dataframe.to_csv(output_folds + '\cross_validation_metrics.csv', index=False)
        print()
        print('Final performance:')
        print()
        print(new_dataframe)        
        print("It is available at", output_folds + '/cross_validation_metrics.csv')

     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_interactions_train', '--input_interactions_train',  
                        help='path of table with tree columns (firts protein A, secund protein B \
                        and third interaction label) e.g., interations_train.txt')
    
    #parser.add_argument('-input_interactions_test', '--input_interactions_test', 
    #                    help='path of table with tree columns (firts protein A, secund protein B \
    #                    and third interaction label) e.g., interations_test.txt')
    
    parser.add_argument('-input_interactions_candidates', '--input_interactions_candidates', help='path of table with two columns (firts protein and secund RNA) of your candidates interactions pairs e.g., candidates.txt', default='')
    
    
    parser.add_argument('-sequences_dictionary', '--sequences_dictionary', help='all sequences in \
                        the problem in fasta format, e.g., dictionary.fasta')
    
    parser.add_argument('-topological_features', '--topological_features', default='yes', help='uses topology features to characterization of the sequences, e.g., yes or no, default=yes)')
    
    parser.add_argument('-output', '--output', help='resutls directory, e.g., result/')
    parser.add_argument('-n_cpu', '--n_cpu', default=1, help='number of cpus - default = 1')
    
    args = parser.parse_args() 
    input_interactions_train = args.input_interactions_train
    #input_interactions_test = args.input_interactions_test
    sequences_dictionary = args.sequences_dictionary
    topological_features = args.topological_features
    foutput = args.output
    n_cpu = args.n_cpu
    
    
    candidates = args.input_interactions_candidates
    if candidates != '':
        candidates = pd.read_csv(candidates, sep=',')
    check_path([input_interactions_train], 'input_interactions_train')          
    
    #if None != input_interactions_test:
    #    check_path([input_interactions_test], 'input_interactions_test')
        
    check_path([sequences_dictionary], 'sequences_dictionary') 
    make_fold(foutput)
    
    feat_eng(input_interactions_train, sequences_dictionary, n_cpu, foutput, candidates, topological_features)
    fit_mod(input_interactions_train, sequences_dictionary, n_cpu, foutput, candidates, topological_features)
    
   