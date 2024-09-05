import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
import networkx as nx

import torch
import statistics
from Bio import SeqIO
import argparse
import subprocess
import sys
import os.path
from subprocess import Popen
from multiprocessing import Manager
from sklearn.exceptions import FitFailedWarning


# Define the reduced alphabet mapping
reduced_alphabet = {
    'A': 0, 'G': 0, 'V': 0,
    'I': 1, 'L': 1, 'F': 1, 'P': 1,
    'Y': 2, 'M': 2, 'T': 2, 'S': 2,
    'H': 3, 'N': 3, 'Q': 3, 'W': 3,
    'R': 4, 'K': 4,
    'D': 5, 'E': 5,
    'C': 6
}

# Function to encode a protein sequence
def encode_protein_sequence(sequence):
    encoded_sequence = np.zeros((7, 7, 7))
    for i in range(len(sequence) - 2):
        triad = sequence[i:i+3]
        if all(aa in reduced_alphabet for aa in triad):
            idx1, idx2, idx3 = [reduced_alphabet[aa] for aa in triad]
            encoded_sequence[idx1, idx2, idx3] += 1

    # Normalize the frequencies
    total_triads = len(sequence) - 2
    encoded_sequence /= total_triads

    # Flatten the 3D matrix into a 1D vector
    return encoded_sequence.flatten()

# Read FASTA file

def group_feats(fasta_file_path, output):
    sequences = []
    for record in SeqIO.parse(fasta_file_path, 'fasta'):
        sequences.append(str(record.seq))

    # Encode each protein sequence
    encoded_sequences = [encode_protein_sequence(seq) for seq in sequences]

    # Create a DataFrame from the encoded sequences
    columns = [f'feature_{i}' for i in range(343)]
    df = pd.DataFrame(encoded_sequences, columns=columns)

    # Add a column for sequence IDs (assuming the FASTA headers contain sequence IDs)
    df['nameseq'] = [record.id for record in SeqIO.parse(fasta_file_path, 'fasta')]

    # Reorder columns with 'nameseq' as the first column
    df = df[['nameseq'] + columns]

    # Save the DataFrame to a CSV file
    csv_file_path = output + '/fq_grups.csv'
    df.to_csv(csv_file_path, index=False)

    #print(f'Data saved to {csv_file_path}')

amino_mappings = {
    "H1": {"A": 0.62, "C": 0.29, "D": -0.9, "E": -0.74, "F": 1.19, "G": 0.48, "H": -0.4, "I": 1.38,
           "K": -1.5, "L": 1.06, "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53, "S": -0.18,
           "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26},
    "H2": {"A": -0.5, "C": -1, "D": 3, "E": 3, "F": -2.5, "G": 0, "H": -0.5, "I": -1.8,
           "K": 3, "L": -1.8, "M": -1.3, "N": 2, "P": 0, "Q": 0.2, "R": 3, "S": 0.3,
           "T": -0.4, "V": -1.5, "W": -3.4, "Y": -2.3},
    "V": {"A": 27.5, "C": 44.6, "D": 40, "E": 62, "F": 115.5, "G": 0, "H": 79, "I": 93.5,
          "K": 100, "L": 93.5, "M": 94.1, "N": 58.7, "P": 41.9, "Q": 80.7, "R": 105, "S": 29.3,
          "T": 51.3, "V": 71.5, "W": 145.5, "Y": 117.3},
    "P1": {"A": 8.1, "C": 5.5, "D": 13, "E": 12.3, "F": 5.2, "G": 9, "H": 10.4, "I": 5.2,
           "K": 11.3, "L": 4.9, "M": 5.7, "N": 11.6, "P": 8, "Q": 10.5, "R": 10.5, "S": 9.2,
           "T": 8.6, "V": 5.9, "W": 5.4, "Y": 6.2},
    "P2": {"A": 0.046, "C": 0.128, "D": 0.105, "E": 0.151, "F": 0.29, "G": 0, "H": 0.23, "I": 0.186,
           "K": 0.219, "L": 0.186, "M": 0.221, "N": 0.134, "P": 0.131, "Q": 0.18, "R": 0.291, "S": 0.062,
           "T": 0.108, "V": 0.14, "W": 0.409, "Y": 0.298},
    "SASA": {"A": 1.181, "C": 1.461, "D": 1.587, "E": 1.862, "F": 2.228, "G": 0.881, "H": 2.025, "I": 1.81,
             "K": 2.258, "L": 1.931, "M": 2.034, "N": 1.655, "P": 1.468, "Q": 1.932, "R": 2.56, "S": 1.298,
             "T": 1.525, "V": 1.645, "W": 2.663, "Y": 2.368},
    "NCI": {"A": 0.007187, "C": -0.03661, "D": -0.02382, "E": 0.006802, "F": 0.037552, "G": 0.179052,
            "H": -0.01069, "I": 0.021631, "K": 0.017708, "L": 0.051672, "M": 0.002683, "N": 0.005392,
            "P": 0.239531, "Q": 0.049211, "R": 0.043587, "S": 0.004627, "T": 0.003352, "V": 0.057004,
            "W": 0.037977, "Y": 0.023599}
}

def fasta_to_dataframe(fasta_file):
    records = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    data = {'Nome da Sequência': list(records.keys()), 'Sequência': [str(record.seq) for record in records.values()]}
    df = pd.DataFrame(data) 
    return df

def prop_qf(data, amino_mappings, path):

    # Criar uma lista para armazenar os dados
    data_list = []

    # Iterar sobre as linhas do DataFrame
    for mapping_name, mapping_values in amino_mappings.items():
        result_df = pd.DataFrame()

        for index, row in data.iterrows():
            name = row[data.columns[0]]
            sequence = row[data.columns[1]]

            sequence_features = []

            numeric_sequence = np.array([mapping_values.get(amino, 0) for amino in sequence])
            fft_result = np.fft.fft(numeric_sequence)

            complex_array = fft_result[:200]

            modulos = np.abs(complex_array)

            num_colunas = len(modulos)  # Obtém o número real de colunas
            df = pd.DataFrame([modulos], columns=[f'{mapping_name}_{i+1}' for i in range(num_colunas)], index=[name])

            # Adicionar o DataFrame ao DataFrame resultante
            result_df = pd.concat([result_df, df])
        result_df = result_df.dropna(axis=1)
        result_df.index.name = 'nameseq'  # Define o nome do índice
        result_df.reset_index(inplace=True)  # Move o índice para uma coluna
        output = os.path.join(path, 'feat_'+mapping_name+'.csv')
        result_df.to_csv(output, index=False)

def prop_qf_mean(data, amino_mappings, path):
    # Criar uma lista para armazenar os dados
    data_list = []

    # Iterar sobre as linhas do DataFrame
    for index, row in data.iterrows():
        name = row[data.columns[0]]
        sequence = row[data.columns[1]]

        sequence_features = []

        for mapping_name, mapping_values in amino_mappings.items():
            numeric_sequence = np.array([mapping_values.get(amino, 0) for amino in sequence])
            fft_result = np.fft.fft(numeric_sequence)
            
            spectrum = np.abs(fft_result) ** 2
            spectrumTwo = np.abs(fft_result)

            # Calcular as características do espectro
            average = np.mean(spectrum)
            median = np.median(spectrum)
            maximum = np.max(spectrum)
            minimum = np.min(spectrum)

            peak = (len(spectrum) / 3) / average
            peak_two = (len(spectrumTwo) / 3) / np.mean(spectrumTwo)
            standard_deviation = np.std(spectrum)
            standard_deviation_pop = statistics.stdev(spectrum)
            percentile15 = np.percentile(spectrum, 15)
            percentile25 = np.percentile(spectrum, 25)
            percentile50 = np.percentile(spectrum, 50)
            percentile75 = np.percentile(spectrum, 75)
            amplitude = maximum - minimum
            variance = statistics.variance(spectrum)
            interquartile_range = np.percentile(spectrum, 75) - np.percentile(spectrum, 25)
            semi_interquartile_range = (np.percentile(spectrum, 75) - np.percentile(spectrum, 25)) / 2
            coefficient_of_variation = standard_deviation / average
            skewness = (3 * (average - median)) / standard_deviation
            kurtosis = (np.percentile(spectrum, 75) - np.percentile(spectrum, 25)) / (2 * (np.percentile(spectrum, 90) - np.percentile(spectrum, 10)))

            # Adicionar as características à lista sequence_features
            sequence_features.extend([
                average, median, maximum, minimum, peak, peak_two, standard_deviation, 
                standard_deviation_pop, percentile15, percentile25, percentile50, 
                percentile75, amplitude, variance, interquartile_range, semi_interquartile_range, 
                coefficient_of_variation, skewness, kurtosis
            ])

        # Adicionar os dados à lista
        data_list.append([name] + sequence_features)

    # Criar os nomes das colunas
    columns = ["nameseq"]
    for mapping_name in amino_mappings.keys():
        features_list = [
            "average", "median", "maximum", "minimum", "peak", "peak_two", "standard_deviation", 
            "standard_deviation_pop", "percentile15", "percentile25", "percentile50", 
            "percentile75", "amplitude", "variance", "interquartile_range", "semi_interquartile_range", 
            "coefficient_of_variation", "skewness", "kurtosis"
        ]
        columns.extend([f"{mapping_name}_{feature}" for feature in features_list])

    df = pd.DataFrame(data_list, columns=columns)
    output = os.path.join(path, 'Mean_feat.csv')
    df.to_csv(output, index=False)


def extrac_math_features(features_amino, sequences, path):
    """
    Extract features from amino acid-based sequences.

    Parameters:
    - features_amino (list): List of feature extraction options.
    - sequences (str): Input sequences in FASTA format.
    - path (str): Output directory path.

    Returns:
    - datasets_extr (list): List of paths to extracted datasets.
    - names_math (list): List of feature names.
    """

    fasta = [sequences]   
    datasets_extr = []
    names_math = []
    commands = []

    print(f'Extracting extructural features with MathFeature...')

    
    """Feature extraction for aminoacids-based sequences"""   
    #print(features_amino)
    for i in range(len(fasta)):
        file = fasta[i].split('/')[-1]
        if i == 0:  # Train
            #preprocessed_fasta = os.path.join(path + '/pre_' + file)
            #subprocess.run(['python', 'other-methods/preprocessing.py',
            #                '-i', fasta[i], '-o', preprocessed_fasta],
            #                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            #sequencias_unicas = {record.seq: record for record in SeqIO.parse(preprocessed_fasta, "fasta")}
            #SeqIO.write(sequencias_unicas.values(), preprocessed_fasta, "fasta")
            preprocessed_fasta = fasta[i]

        if 1 in features_amino:
            dataset = os.path.join(path, 'Shannon' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('Shannon')
            commands.append(['python', 'MathFeature/methods/EntropyClass.py',
                            '-i', preprocessed_fasta, '-o', dataset, '-l', 'protein',
                            '-k', '5', '-e', 'Shannon'])

        if 2 in features_amino:
            dataset = os.path.join(path, 'Tsallis_23'  + '.csv')
            datasets_extr.append(dataset)
            names_math.append('Tsallis_23')
            commands.append(['python', 'other-methods/TsallisEntropy.py',
                            '-i', preprocessed_fasta, '-o', dataset, '-l', 'protein',
                            '-k', '5', '-q', '2.3'])

        if 3 in features_amino:
            dataset = os.path.join(path, 'Tsallis_30'  + '.csv')
            datasets_extr.append(dataset)
            names_math.append('Tsallis_30')
            commands.append(['python', 'other-methods/TsallisEntropy.py',
                            '-i', preprocessed_fasta, '-o', dataset, '-l', 'protein',
                            '-k', '5', '-q', '3.0'])

        if 4 in features_amino:
            dataset = os.path.join(path, 'Tsallis_40'  + '.csv')
            datasets_extr.append(dataset)
            names_math.append('Tsallis_40')
            commands.append(['python', 'other-methods/TsallisEntropy.py',
                            '-i', preprocessed_fasta, '-o', dataset, '-l', 'protein',
                            '-k', '5', '-q', '4.0'])

        if 5 in features_amino:
            dataset = os.path.join(path, 'ComplexNetworks' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('ComplexNetworks')
            commands.append(['python', 'MathFeature/methods/ComplexNetworksClass-v2.py', '-i',
                            preprocessed_fasta, '-o', dataset, '-l', 'protein',
                            '-k', '3'])

        if 6 in features_amino:
            dataset_di = os.path.join(path, 'kGap_di' + '.csv')
            datasets_extr.append(dataset_di)
            names_math.append('kGap_di')
            commands.append(['python', 'MathFeature/methods/Kgap.py', '-i',
                            preprocessed_fasta, '-o', dataset_di, '-l',
                            'protein', '-k', '1', '-bef', '1',
                            '-aft', '1', '-seq', '3'])

        if 7 in features_amino:
            dataset = os.path.join(path, 'AAC' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('AAC')
            commands.append(['python', 'MathFeature/methods/ExtractionTechniques-Protein.py', '-i',
                            preprocessed_fasta, '-o', dataset, '-l', 'protein','-t', 'AAC'])

        if 8 in features_amino:
            dataset = os.path.join(path, 'DPC' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('DPC')
            commands.append(['python', 'MathFeature/methods/ExtractionTechniques-Protein.py', '-i',
                            preprocessed_fasta, '-o', dataset, '-l', 'protein', '-t', 'DPC'])

        if '9' in features_amino:
            dataset = os.path.join(path, 'TPC' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('TPC')
            commands.append(['python', 'MathFeature/methods/ExtractionTechniques-Protein.py', '-i',
                            preprocessed_fasta, '-o', dataset, '-l', 'protein', '-t', 'TPC'])

        if '10' in features_amino:
            dataset = os.path.join(path, 'iFeature' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('iFeature')
            commands.append(['python', 'other-methods/iFeature-modified/iFeature.py', '--file', 
                             preprocessed_fasta, '--type', 'All', '--label', 'protein', 
                             '--out', dataset])

        if '11' in features_amino:
            dataset = os.path.join(path, 'Global' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('Global')
            commands.append(['python', 'other-methods/modlAMP-modified/descriptors.py', '-option',
                             'peptide', '-label', 'protein', '-input', preprocessed_fasta, 
                             '-output', dataset])

        if '12' in features_amino:
            dataset = os.path.join(path, 'Peptide' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('Peptide')
            commands.append(['python', 'other-methods/modlAMP-modified/descriptors.py',
                             '-option','peptide', '-label', 'protein', '-input', 
                             preprocessed_fasta, '-output', dataset])    
            
        if 9 in features_amino:
            dataset = os.path.join(path, 'Mean_feat' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('Mean_feat')
            df = fasta_to_dataframe(preprocessed_fasta)
            prop_qf(df,amino_mappings, path)

        if '10' in features_amino:
            dataset = os.path.join(path, 'feat_H1' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('feat_H1')
            dataset = os.path.join(path, 'feat_H2' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('feat_H2')
            dataset = os.path.join(path, 'feat_NCI' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('feat_NCI')
            dataset = os.path.join(path, 'feat_P1' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('feat_P1')
            dataset = os.path.join(path, 'feat_P2' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('feat_P2')
            dataset = os.path.join(path, 'feat_SASA' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('feat_SASA')
            dataset = os.path.join(path, 'feat_V' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('feat_V')
            df = fasta_to_dataframe(preprocessed_fasta)
            prop_qf_mean(df, amino_mappings, path)

        if 11 in features_amino:            
            dataset = os.path.join(path, 'fq_grups' + '.csv')
            datasets_extr.append(dataset)
            names_math.append('fq_grups')
            group_feats(preprocessed_fasta, path)

            
        """Concatenating all the extracted features"""
        processes = [Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) for cmd in commands]
        for p in processes: p.wait()   
    return datasets_extr, names_math

def graph_ext(G):
    """
    Extract various centrality and graph measures from a given graph.

    Parameters:
        G (networkx.Graph): Input graph.

    Returns:
        data_nodes (dict): Dictionary containing extracted centrality and graph measures.
    """
    hits_scores = nx.hits(G)
    hubs = hits_scores[0]
    authorities = hits_scores[1]
    adj_matrix = nx.adjacency_matrix(G).toarray()
    spectrum = np.linalg.eigvals(adj_matrix)
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    #eigenvector_centrality = nx.eigenvector_centrality(G)
    pagerank = nx.pagerank(G)
    harmonic_centrality = nx.harmonic_centrality(G)
    personalized_pagerank = nx.pagerank(G, alpha=0.5)
    directed_eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight', max_iter=10)
    load_centrality = nx.load_centrality(G)
    closeness_in_closeness_centrality = nx.closeness_centrality(G, wf_improved=False)
    adjusted_closeness_centrality = nx.closeness_centrality(G, wf_improved=False, distance="adjusted")
    weighted_closeness_centrality = nx.closeness_centrality(G, wf_improved=False, distance="weighted")
    edge_betweenness = nx.edge_betweenness_centrality(G)
    load_centrality = nx.load_centrality(G)
    #subgraph_centrality = nx.subgraph_centrality(G)
    #information_centrality = nx.information_centrality(G)
    directed_eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight', max_iter=10)
    #current_flow_betweenness = nx.current_flow_betweenness_centrality(G)
    #current_flow_closeness = nx.current_flow_closeness_centrality(G)
    closeness_in_closeness_centrality = nx.closeness_centrality(G, wf_improved=False)
    adjusted_closeness_centrality = nx.closeness_centrality(G, wf_improved=False, distance="adjusted")
    weighted_closeness_centrality = nx.closeness_centrality(G, wf_improved=False, distance="weighted")
    #eccentricity_centrality = nx.eccentricity(G)
    harmonic_centrality = nx.harmonic_centrality(G)
    average_neighbor_degree_centrality = nx.average_neighbor_degree(G)
    degree_assortativity_centrality = nx.degree_assortativity_coefficient(G)
    clustering_centrality = nx.clustering(G)
    core_number_centrality = nx.core_number(G)

    data_nodes = {
        'Node': list(G.nodes()),
        'Hub Score': [hubs[node] for node in G.nodes()],
        'Authority Score': [authorities[node] for node in G.nodes()],
        'Spectrum': spectrum,
        'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
        'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes()],
        'Closeness Centrality': [closeness_centrality[node] for node in G.nodes()],
        #'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes()],
        'PageRank': [pagerank[node] for node in G.nodes()],
        'Harmonic Centrality': [harmonic_centrality[node] for node in G.nodes()],
        'Personalized PageRank (Alpha=0.5)': [personalized_pagerank[node] for node in G.nodes()],
        'Closeness in Closeness Centrality': [closeness_in_closeness_centrality[node] for node in G.nodes()],
        'Adjusted Closeness Centrality': [adjusted_closeness_centrality[node] for node in G.nodes()],
        'Weighted Closeness Centrality': [weighted_closeness_centrality[node] for node in G.nodes()],
        #'Eccentricity Centrality': [eccentricity_centrality[node] for node in G.nodes()],
        'Average Neighbor Degree Centrality': [average_neighbor_degree_centrality[node] for node in G.nodes()],
        'Clustering Centrality': [clustering_centrality[node] for node in G.nodes()],
        'Core Number Centrality': [core_number_centrality[node] for node in G.nodes()],
        #'Information Centrality': [information_centrality[node] for node in G.nodes()],
        'Directed Eigenvector Centrality': [directed_eigenvector_centrality[node] for node in G.nodes()],
        #'Current Flow Betweenness': [current_flow_betweenness[node] for node in G.nodes()],
        #'Closeness in Closeness Centrality': [closeness_in_closeness_centrality[node] for node in G.nodes()],
        'Adjusted Closeness Centrality': [adjusted_closeness_centrality[node] for node in G.nodes()],
        'Weighted Closeness Centrality': [weighted_closeness_centrality[node] for node in G.nodes()],
        #'Eccentricity Centrality': [eccentricity_centrality[node] for node in G.nodes()],
        'Harmonic Centrality': [harmonic_centrality[node] for node in G.nodes()],
        'Clustering Centrality': [clustering_centrality[node] for node in G.nodes()],
        'Core Number Centrality': [core_number_centrality[node] for node in G.nodes()]
        #'Subgraph Centrality': [subgraph_centrality[node] for node in G.nodes()]
    }
    return data_nodes

def make_graph(edges, node):
    
    #print(f'Extracting topological features...')
    
    """
    Create a graph from edges data and nodes data.

    Parameters:
        edges (pd.DataFrame): DataFrame containing edge information.
        node (pd.DataFrame): DataFrame containing node information.

    Returns:
        G (nx.Graph): NetworkX Graph representing the connections between nodes.
    """
    # Extract unique protein names from node data
    columns_graph = edges.columns
    PA = node[columns_graph[0]].unique()
    PB = node[columns_graph[1]].unique()
    
    # Create an empty undirected graph
    G = nx.Graph()
    G.add_nodes_from(PA)
    G.add_nodes_from(PB)
    
    # Add edges to the graph based on edge data
    for index, row in edges.iterrows():
        if row[columns_graph[2]] == 1:
            protein_a = row[columns_graph[0]]
            protein_b = row[columns_graph[1]]
            G.add_edge(protein_a, protein_b)
    
    # Remove self-loops from the graph
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    
    return G

def make_graph_all(edges, node):
    
    #print(f'Extracting topological features...')
        
    """
    Create a graph from edges data with both positive and negative edges and nodes data.

    Parameters:
        edges (pd.DataFrame): DataFrame containing edge information.
        node (pd.DataFrame): DataFrame containing node information.

    Returns:
        G (nx.Graph): NetworkX Graph representing the connections between nodes.
    """
    # Extract unique protein names from node data
    columns_graph = edges.columns
    PA = node[columns_graph[0]].unique()
    PB = node[columns_graph[1]].unique()
    
    # Create an empty undirected graph
    G = nx.Graph()
    G.add_nodes_from(PA)
    G.add_nodes_from(PB)
    
    # Add edges to the graph based on edge data (including positive and negative edges)
    for index, row in edges.iterrows():
        if row[columns_graph[2]] == 1 or row[columns_graph[2]] == 0:
            protein_a = row[columns_graph[0]]
            protein_b = row[columns_graph[1]]
            G.add_edge(protein_a, protein_b)
    
    # Remove self-loops from the graph
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    
    return G