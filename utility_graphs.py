import logging
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from itertools import zip_longest
from os.path import join, basename, exists
from sys import stdout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
import networkx as nx
import statistics
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score, f1_score,
    auc, accuracy_score, balanced_accuracy_score, roc_auc_score
)

report_handler = logging.StreamHandler(stream=stdout)
report_handler.setLevel(logging.WARNING)
report_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))

report_logger = logging.getLogger(basename(__file__))
report_logger.addHandler(report_handler)



REPORT_USABILITY_TITLE_BINARY = "Model Interpretability Report (BioPrediction)"
REPORT_USABILITY_DESCRIPITION = "This a usability report for explain some models aspects that can be util for your aplication."

REPORT_1 = """
Type one ...
"""

REPORT_2 = """
Type two ...
"""

REPORT_3 = """
Type tree ...
"""


def final_predictions(X_test, y_test, y_pares, model, output):  
    """
    Generate final predictions using a machine learning model, append them to a DataFrame, and save the results to a CSV file.

    Parameters:
    - X_test: The test data for which predictions are generated.
    - y_test: The true labels or target values associated with the test data.
    - y_pares: A DataFrame with additional information (e.g., identifiers) related to the predictions.
    - model: The machine learning model used for making predictions.
    - output: The directory where the CSV file with final predictions will be saved.

    Returns:
    - output_data: A DataFrame containing the final predictions and related information.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    output_data = y_pares
    output_data['Class_pred'] = y_pred
    output_data['Class_real'] = y_test
    output_data[['Probability of class 0', 'Probability of class 1']] = pd.DataFrame(y_proba)
    output_data = output_data.drop('Probability of class 0', axis=1)
    output_data = output_data.sort_values(by='Probability of class 1', ascending=False)
    output_data.to_csv(output+'/list_final_predictions.csv', index=False)
    return output_data



def dist_score(output_data, output):
    output_data[['ProteinA', 'ProteinB']] = output_data['nameseq'].str.split('_', expand=True)
    df = output_data
    # Filter the data for positive and negative interactions
    df_positive = df[df['Class_real'] == 1]
    df_negative = df[df['Class_real'] == 0]

    print(f"Number of negative interactions: {len(df_negative)}")

    # Create histogram with both labels
    plt.figure(figsize=(8, 5))
    plt.hist(df_positive['Probability of class 1'], bins=20, color='#A1CAF1', edgecolor='black', alpha=0.7, label='Label 1 (Positive)')
    plt.hist(df_negative['Probability of class 1'], bins=20, color='#FFABAB', edgecolor='black', alpha=0.7, label='Label 0 (Negative)')

    # Graph settings
    plt.xlabel('Pairs scores')
    plt.ylabel('Number of interactions')
    plt.title('Distribution of scores for the entire test set')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Limit the Y-axis to 1 time the number of positive interactions
    plt.ylim(0, 1 * len(df_positive))

    # Save the graph
    graph_path = f"{output}/precision_vs_threshold_graph.png"
    plt.savefig(graph_path)

    return graph_path




def precision_graph_int(output_data, output):
    """
    Generate and save a precision vs. threshold graph using filtered data.

    Parameters:
    - output_data: A DataFrame containing the final predictions and related information.
    - output: The directory where the graph image will be saved.

    Returns:
    - output: The file path to the saved precision vs. threshold graph image.
    """
    # Separar 'nameseq' em 'ProteinA' e 'ProteinB'
    output_data[['ProteinA', 'ProteinB']] = output_data['nameseq'].str.split('_', expand=True)

    # Definir os intervalos
    score1 = [1.00, 0.90, 0.80, 0.70, 0.60]
    score2 = [0.90, 0.80, 0.70, 0.60, 0.499]

    intervals = []
    precisions = []
    interaction_counts = []

    # Definir cores pastéis para cada intervalo
    colors = ['#A1CAF1', '#B2D8B2', '#FFD59A', '#FFABAB', '#D7BDE2']
    # Calcular a precisão para cada intervalo
    for i in range(len(score1)):
        filtered_df = output_data[(output_data['Probability of class 1'] <= score1[i]) & (output_data['Probability of class 1'] > score2[i])]
        
        n_interactions = len(filtered_df)
        if n_interactions > 0:
            precision = len(filtered_df[filtered_df['Class_real'] == 1]) / n_interactions
        else:
            precision = 0  # Evita divisão por zero

        # Salvar para o gráfico com intervalo invertido
        interval_label = f"{score1[i]} - {score2[i]}"
        intervals.append(interval_label)
        precisions.append(precision)
        interaction_counts.append(n_interactions)

        #print(f"Interval {interval_label}: {precision:.4f}, Nº interactions: {n_interactions}")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(intervals, precisions, color=colors, alpha=0.8, edgecolor='black')

    plt.xlabel('Pairs scores Intervals')
    plt.ylabel('Precision')
    plt.title('Precision in different Candidates scores intervals for test interactions')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Add precision values on the bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, 
                 f"{precisions[i]:.2f}", ha='center', fontsize=10)

    # Create detailed legend with the same colors as the chart
    legend_labels = [f"{intervals[i]}: {interaction_counts[i]} interactions" for i in range(len(intervals))]
    legend_patches = [plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.8, edgecolor='black') for i in range(len(intervals))]
    plt.legend(legend_patches, legend_labels, loc='upper right', fontsize=8, title="Nº of Interactions")

    # Save the graph
    graph_path = f"{output}/precision_vs_threshold(intervals)_graph.png"
    plt.savefig(graph_path)


    return graph_path



def precision_graph(output_data, output):
    """
    Generate and save a precision vs. threshold graph using filtered data.

    Parameters:
    - output_data: A DataFrame containing the final predictions and related information.
    - output: The directory where the graph image will be saved.

    Returns:
    - output: The file path to the saved precision vs. threshold graph image.
    """
    generated_plt = []
 
    # Set the initial threshold value for filtering data.
    threshold = 0.5
    filtered_data = output_data[output_data['Probability of class 1'] > threshold]
    y_true_filtered = filtered_data['Class_real']
    y_pred_filtered = filtered_data['Class_pred']
    filtered_data.sort_values(by='Probability of class 1', ascending=False, inplace=True)

    # Define a range of thresholds for analysis.
    thresholds = np.arange(0.99, 0.45, -0.05)
    accuracies = []

    for threshold in thresholds:
        threshold_data = filtered_data[filtered_data['Probability of class 1'] >= threshold]
        accuracy = precision_score(threshold_data['Class_real'], threshold_data['Class_pred'])
        accuracies.append(accuracy)
    thresholds = thresholds[::-1]
    accuracies = accuracies[::-1]

    plt.plot(thresholds, accuracies, marker='o', linestyle='-')
    plt.title('Precision vs. Threshold (Threshold > 50%)')
    plt.xlabel('Threshold (%)')
    plt.ylabel('Precision')
    plt.gca().invert_xaxis()  # Invert the x-axis for increasing thresholds.
    plt.grid(True)

    output = output + '/graph_precision.png'
    plt.savefig(output, format='png')  # dpi=300, bbox_inches='tight')
    plt.clf()
    
    # Return the file path to the saved precision vs. threshold graph.
    return output


def coverage_graph(output_data, output):
    """
    Generate and save a recall vs. threshold graph.

    Parameters:
    - output_data: A DataFrame containing the final predictions and related information.
    - output: The directory where the graph image will be saved.

    Returns:
    - output: The file path to the saved recall vs. threshold graph.
    """
    
    y_true = output_data['Class_real']
    y_score = output_data['Probability of class 1']
    
    thresholds = np.sort(y_score.unique())[::-1]  # Ordena os thresholds em ordem decrescente
    recalls = []
    
    total_positives = sum(y_true)
    
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        true_positives = sum((y_pred == 1) & (y_true == 1))
        recall = true_positives / total_positives if total_positives > 0 else 0
        recalls.append(recall)
    
    plt.plot(thresholds, recalls)
    plt.xlabel('Threshold (%)')
    plt.ylabel('Recall')
    plt.title('Recall vs. Threshold')
    plt.axvline(x=0.5, color='red', linestyle='--', label='conventional division')
    plt.gca().invert_xaxis()  # Invert the x-axis for increasing thresholds.
    plt.grid(True)
    plt.legend()
    
    output_path = output + '/graph_recall.png'
    plt.savefig(output_path, format='png')
    
    return output_path


def make_fig_graph(output_data, output):
    """
    Create and save graphs based on node connections using NetworkX.

    Parameters:
    - output_data: A DataFrame containing information, including 'nameseq', 'Class_pred', etc.
    - output: The directory where the generated graphs will be saved.

    Returns:
    - None
    """
    # Extract 'nameseq' values for sequences predicted as class 1.
    names = output_data.loc[output_data['Class_pred'] == 1, 'nameseq'].tolist()

    aminoA, aminoB = [], []
    for i in names:
        nu, am = i.split("_", 1)
        aminoA.append(nu)
        aminoB.append(am)

    # Create an undirected graph using NetworkX.
    G = nx.Graph()

    # Sort the lists by length to ensure consistent node order.
    list1 = sorted(aminoA, key=len)
    list2 = sorted(aminoB, key=len)

    # Add nodes to the graph.
    G.add_nodes_from(list1)
    G.add_nodes_from(list2)

    # Add edges to connect nodes.
    for i in range(len(list1)):
        G.add_edge(list1[i], list2[i])

    # Calculate degrees for each node.
    d = dict(G.degree())

    # Define threshold degrees for selecting nodes.
    threshold_degrees = [0.8, 0.15, 0.05]
    deg = []

    for k in threshold_degrees:
        sorted_degrees = sorted(d.values(), reverse=True)
        num_top_nodes = int(len(sorted_degrees) * k)
        grau_10_percent = sorted_degrees[num_top_nodes - 1]
        deg.append(grau_10_percent)

        

        
    for j in range(3):
        d = dict(G.degree())
        # Filter nodes based on degree threshold.
        filtered_nodes = [node for node, degree in d.items() if degree >= deg[j]]
        G_ = G.subgraph(filtered_nodes)
        d = dict(G_.degree())
        
        
        node_colors = []
        for node in G_.nodes():
            if node in list1:
                node_colors.append('skyblue')  # Azul para nós em list1
            elif node in list2:
                node_colors.append('orange')  # Laranja para nós em list2
            #else:
                #node_colors.append('skyblue')  # Cor padrão para outros nós

        # Create a graph visualization.
        pos = nx.spring_layout(G_)
        fig, ax = plt.subplots(figsize=(28 * 0.5 *3, 21 * 0.75 * 3))
        nx.draw(G_, node_size=[v * 30 for v in d.values()], pos=pos, node_color=node_colors, ax=ax)
        nx.draw_networkx_edges(G_, pos, width=2, edge_color='gray', ax=ax)
        labels = {node: node for node in G_.nodes()}
        nx.draw_networkx_labels(G_, pos, labels, font_size=8, font_color='black', font_family='sans-serif', ax=ax, font_weight='bold')
        ax.set_axis_off()
        plt.tight_layout()

        # Save the graph image.
        output_name = output + '/graph_degree_' + str(deg[j]) + 'or_more.png'
        plt.savefig(output_name, format='png')
        plt.clf()
 
