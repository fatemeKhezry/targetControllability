from unittest import result
import pandas as pd
import numpy as np
import networkx as nx
import random as rand
import copy
import openpyxl
import itertools
from networkx.algorithms import bipartite
from numpy.linalg import matrix_power
from itertools import chain
from pyvis.network import Network
from openpyxl import load_workbook

def read_adj_matrix(address):
    """
    Reads an adjacency matrix from an Excel file.
    
    Parameters:
    address (str): Path to the Excel file containing the adjacency matrix.
    
    Returns:
    numpy.ndarray: Adjacency matrix as a numpy array.
    """
    adj_matrix = pd.read_excel(address, header=None)
    return adj_matrix.to_numpy().astype(int).T

def edges_list_to_graph(address):
    """
    Converts a list of edges from an Excel file to a directed graph.
    
    Parameters:
    address (str): Path to the Excel file containing the edge list.
    
    Returns:
    networkx.DiGraph: Directed graph created from the edge list.
    """
    e = pd.read_excel(address, header=None).to_numpy()
    nodes = list(set(e[:, 0]).union(set(e[:, 1])))
    edges = list({tuple(item) for item in e})
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def target_matrix(target_index, adj_matrix):
    """
    Creates a target matrix for the given target indices.
    
    Parameters:
    target_index (list): List of target indices.
    adj_matrix (numpy.ndarray): Adjacency matrix.
    
    Returns:
    numpy.ndarray: Target matrix.
    """
    matrix = np.zeros((len(target_index), len(adj_matrix))).astype(int)
    for i in range(len(target_index)):
        matrix[i, target_index[i]] = 1
    return matrix

def driver_matrix(network, driver_index):
    """
    Creates a driver matrix for the given driver indices.
    
    Parameters:
    network (numpy.ndarray): The network adjacency matrix.
    driver_index (list): List of driver indices.
    
    Returns:
    numpy.ndarray: Driver matrix.
    """
    matrix = np.zeros((len(network), len(driver_index))).astype(int)
    for i in range(len(driver_index)):
        matrix[driver_index[i], i] = 1
    return matrix

def kalman(A, driver_index, target_index, maxpath=10):
    """
    Kalman filter function for control theory applications.
    
    Parameters:
    A (numpy.ndarray): Adjacency matrix.
    driver_index (list): List of driver indices.
    target_index (list): List of target indices.
    maxpath (int): Maximum path length.
    
    Returns:
    list: List of matrices for each path length.
    """
    n_matrixes = [None] * maxpath
    n_matrixes[0] = np.identity(len(A))[target_index, :].astype(object)
    n_matrixes[0] = n_matrixes[0][:, driver_index]
    A_power = np.identity(len(A))
    for i in range(1, maxpath):
        A_power = A_power @ A
        n_matrixes[i] = A_power[target_index, :][:, driver_index]
    return n_matrixes

def find_matrix_base(M):
    """
    Finds the base of the given matrix using Gaussian elimination.
    
    Parameters:
    M (numpy.ndarray): The matrix.
    
    Returns:
    list: List of independent columns.
    """
    rank = np.linalg.matrix_rank(M)
    _, _, v = np.linalg.svd(M)
    return [i for i in range(len(v)) if np.abs(v[i]).sum() > 1e-10][:rank]


def optimize_drivers(finaldriver, controlled_target_index, not_controlled, AdjMatrix, targetlist):
    """
    Optimizes the set of driver nodes to control the network.
    
    Parameters:
    finaldriver (list): Initial list of driver indices.
    controlled_target_index (list): List of controlled target indices.
    not_controlled (list): List of target indices that are not yet controlled.
    AdjMatrix (numpy.ndarray): Adjacency matrix.
    targetlist (list): List of target indices.
    
    Returns:
    list: Optimized list of driver indices.
    """
    target_size = len(targetlist)
    final_rank = len(controlled_target_index)
    
    while final_rank < target_size:
        newdriver = None
        max_rank = final_rank
        
        for i in not_controlled:
            tdriver = finaldriver.copy()
            tdriver.append(i)
            tkalman = kalman(AdjMatrix, tdriver, targetlist)
            temp_rank = len(find_matrix_base(tkalman))
            
            if temp_rank > max_rank:
                newdriver = i
                max_rank = temp_rank
        
        final_rank = max_rank
        
        if newdriver is None:
            finaldriver.extend(not_controlled)
            break
        
        finaldriver.append(newdriver)
    
    return finaldriver





def N_matrixes(A, targetIndex, maxpath=10):
    """
    Computes a list of matrices, where each matrix is a power of the adjacency matrix A 
    limited to the rows specified by targetIndex.
    
    Parameters:
    A (numpy.ndarray): Adjacency matrix.
    targetIndex (list): List of target indices.
    maxpath (int, optional): Maximum path length. Default is 10.
    
    Returns:
    list: List of matrices, where the i-th matrix is A^i[targetIndex, :].
    """
    A = A.astype(object)
    n_matrixes = [None] * maxpath
    n_matrixes[0] = np.identity(len(A))[targetIndex, :].astype(object)
    n_matrixes[1] = A[targetIndex, :].astype(object)
    
    for i in range(2, maxpath):
        n_matrixes[i] = matrix_power(A, i)[targetIndex, :].astype(object)
    
    return n_matrixes


def find_minimum_rank_matrixes(matrixes):
    """
    Finds the indices of matrices with the minimum non-zero rank from a list of matrices.
    
    Parameters:
    matrixes (list): List of numpy.ndarray matrices.
    
    Returns:
    list: Indices of matrices with the minimum non-zero rank.
    """
    minimum_rank = len(matrixes)
    min_rank_matrixes = []  # Indices of matrices with minimum rank
    
    for i in range(len(matrixes)):
        rank = len(find_matrix_base(matrixes[i]))
        if 0 < rank < minimum_rank:
            min_rank_matrixes = [i]
            minimum_rank = rank
        elif rank == minimum_rank:
            min_rank_matrixes.append(i)
    
    return min_rank_matrixes

def create_one_driver_kalman(d, matrixes):
    """
    Creates a Kalman matrix for a single driver based on the given list of matrices.
    
    Parameters:
    d (int): Index of the driver.
    matrixes (list): List of numpy.ndarray matrices, as output by the N_matrixes function.
    
    Returns:
    numpy.ndarray: Kalman matrix for the specified driver.
    """
    kalman = np.zeros((len(matrixes[0]), len(matrixes))).astype(object)
    for m in range(len(matrixes)):
        kalman[:, m] = matrixes[m][:, d]
    return kalman

def best_drivers(bases, matrixes):
    """
    Finds the best driver nodes that maximize the rank of the Kalman matrix.
    
    Parameters:
    bases (list): List of potential driver indices.
    matrixes (list): List of numpy.ndarray matrices, as output by the N_matrixes function.
    
    Returns:
    list: Indices of the best driver nodes.
    """
    maxrank = 0
    best_nodes = []
    
    for i in bases:
        kalman = create_one_driver_kalman(i, matrixes)
        kalman_rank = len(find_matrix_base(kalman))
        if kalman_rank > maxrank:
            maxrank = kalman_rank
            best_nodes = [i]
        elif kalman_rank == maxrank:
            best_nodes.append(i)
    
    return best_nodes


def max_control(min_rank_matrixes, matrixes):
    """
    Determines the maximum control set and best driver node from the minimum rank matrices.
    
    Parameters:
    min_rank_matrixes (list): List of indices of matrices with the minimum non-zero rank.
    matrixes (list): List of numpy.ndarray matrices, as output by the N_matrixes function.
    
    Returns:
    tuple: A tuple containing:
        - list: Indices of independent rows in the best Kalman matrix.
        - int: Index of the best driver node.
    """
    linearIndependentColumn = []

    # Collect all bases from the transpose of the minimum rank matrices
    for i in min_rank_matrixes:
        bases = find_matrix_base(matrixes[i].T)
        linearIndependentColumn.extend(bases)
    
    # Remove duplicate columns
    linearIndependentColumn = list(set(linearIndependentColumn))
    
    # Find the best driver nodes
    control = best_drivers(linearIndependentColumn, matrixes)
    
    # Create the best Kalman matrix and find the best driver
    bestdriver = control[-1]
    bestkalman = create_one_driver_kalman(bestdriver, matrixes)
    
    # Find the independent rows in the best Kalman matrix
    independent_rows = find_matrix_base(bestkalman)
    
    return independent_rows, bestdriver


def read_target_list(address):
    """
    Reads a target list from an Excel file.
    
    Parameters:
    address (str): Path to the Excel file containing the target list.
    
    Returns:
    numpy.ndarray: Target list as a numpy array.
    """
    target_table = pd.read_excel(address, header=None).to_numpy()
    return target_table




def refine_drivers(finaldriver, Mtrcs, final_rank):
    """
    Refines the set of driver nodes by removing redundant drivers.
    
    Parameters:
    finaldriver (list): List of driver indices.
    MS (list): List of matrices, as output by the N_matrixes function.
    final_rank (int): Final rank of the Kalman matrix.
    
    Returns:
    list: Refined list of driver indices.
    """
    for i in finaldriver.copy():
        tempdriverset = finaldriver.copy()
        tempdriverset.remove(i)
        
        if tempdriverset:
            temp_kalman = Mtrcs[0][:, tempdriverset]
            for j in range(1, len(Mtrcs)):
                tmp = Mtrcs[j][:, tempdriverset].copy()
                temp_kalman = np.concatenate((temp_kalman, tmp), axis=1)
                
            if len(find_matrix_base(temp_kalman)) == final_rank:
                finaldriver.remove(i)
    
    return finaldriver



networkAdd = 'toynetwork.xlsx'
targetAdd = 'Target.xlsx'  # Insert your target set path

# Read the adjacency matrix from the Excel file
AdjMatrix = read_adj_matrix(networkAdd)

# Read the target list from the Excel file
targetList = read_target_list(targetAdd)

# Create a list of nodes based on the size of the adjacency matrix
nodes = list(range(len(AdjMatrix)))



maxpath = 20
if len(AdjMatrix) < maxpath:
    maxpath = len(AdjMatrix)


# Identify targets that cannot be driven by any nodes except themselves
target_in_degree_zeros = set()
for i in targetList:
    if np.sum(AdjMatrix[:, i]) < 1:
        target_in_degree_zeros.add(i)


# Create the target matrix for the given target indices
targetmatrix = target_matrix(targetList, AdjMatrix)

# Compute the list of matrices using the N_matrixes function
MS = N_matrixes(AdjMatrix, targetList, maxpath)
matrixes = copy.deepcopy(MS)

drivers = []
network_nodes= list(range(len(AdjMatrix)))


if target_in_degree_zeros:
    klmn = kalman(AdjMatrix, list(target_in_degree_zeros), targetList, maxpath)
    l = find_matrix_base(klmn)
    rrows = [r for r in range(len(targetList)) if r not in l]
    for i in range(len(matrixes)):
        matrixes[i] = matrixes[i][rrows, :]
        rcols = [c for c in range(len(matrixes[0][0])) if c not in target_in_degree_zeros]
    for i in range(len(matrixes)):
        matrixes[i] = matrixes[i][:, rcols]
    drivers = list(target_in_degree_zeros)
    network_nodes = [item for item in network_nodes if item not in drivers]


while len(matrixes[0])!=0:
        minrankmatrixes = find_minimum_rank_matrixes(matrixes)
        driver_and_targets = max_control(minrankmatrixes,matrixes)
        rrows = [item for item in range(len(matrixes[0][:,0])) if item not in driver_and_targets[0]]
        rcols = list(range(len(matrixes[0][0,:])))
        rcols.pop(driver_and_targets[1])
        for i in range(len(matrixes)):
            matrixes[i]=matrixes[i][:,rcols]
            matrixes[i] = matrixes[i][rrows,:]
        drivers.append(network_nodes.pop(driver_and_targets[1]))

klmn = kalman(AdjMatrix , drivers , targetList)

controled_target_index= find_matrix_base(klmn)

targetmatrix = target_matrix(targetList, AdjMatrix)

controlled_target = set()
for i in controled_target_index:
    controlled_target.add(np.nonzero(targetmatrix[i])[0][0])
not_controlled = list(set(targetList)-controlled_target)

optimized_drivers = optimize_drivers(drivers, controlled_target, not_controlled, AdjMatrix, targetList)
C = kalman( AdjMatrix , drivers, optimized_drivers)
final_rank = len(find_matrix_base(C))
final_result = refine_drivers(drivers, MS, final_rank)

