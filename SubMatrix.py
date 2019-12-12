
# coding: utf-8

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import scipy as sc
import csv
import copy

def init_X0(len_A, p):

    n = len_A
    I = np.eye(p)
    A = np.arange(n, (int(n/p) + 1)*p)
    B = np.tile(I, (int(n/p)+1, 1))
    X0 = np.delete(B, A, axis = 0)

    return X0

def Eig_matrix(A, X, p):
    C = np.dot(np.dot(X.T, A), X)
    lmda, Q = np.linalg.eig(C)
    print (lmda)
    print (Q)
    return lmda, Q

def Eig_matrix_2(A, B):
    eig_val,eig_vec =  sc.linalg.eig(A,B)
    for i in range(len(eig_vec)): #正規化
        eig_vec [i] = eig_vec[i]/np.linalg.norm(eig_vec[i])

    return eig_val, eig_vec

def Subspace(A, B, n):
    p = n + int(n/5)
    X0 = init_X0(len(A), p)
    I = np.eye(p)
    lmda_before = np.ones(p)
    SS_Flag = True
    _k = 0

    while SS_Flag:
        Z_ = np.dot(B, X0)
        Y = np.dot( np.linalg.inv(A), Z_)

        A_new = np.dot(np.dot(Y.T, A), Y)
        B_new = np.dot(np.dot(Y.T, B), Y)

        lmda, Q = Eig_matrix_2(A_new, B_new)

        X0 = np.dot(Y, Q)
        # lmda_error = np.sum(lmda_before - lmda)
        # if (lmda_error < 1e-10):
            # SS_Flag = False
        if (_k > 20):
            SS_Flag = False
        
        _k += 1
        # lmda_before = copy.deepcopy(lmda)

        print ("Lmda >> ")
        print (lmda)

    return lmda

if __name__ == "__main__":
    A = [[3, 0, 0, 0, 0], 
        [0, -1, 0, 0, 0], 
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0.5, 0.5], 
        [0, 0, 0, -0.5, 0.5]]

    B = [[2, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0], 
        [0, 0, 0.5, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]]


    Lmda = Subspace(A, B, 6)

    print ("Lmda >> ")
    print (Lmda)
    # n = 1
    # p = 4 # 取得する固有振動数

    # # A = [[1, 2, 3, 3, 2], 
    # #     [2, 2, 3, 4, 4], 
    # #     [3, 3, 3, 4, 5],
    # #     [3, 4, 4, 4, 5], 
    # #     [2, 4, 5, 5, 6]]

    # # B = [[1, 2, 2, 3, 3], 
    # #     [2, 2, 3, 3, 4], 
    # #     [2, 3, 3, 4, 4],
    # #     [3, 3, 4, 4, 5],
    # #     [3, 4, 4, 5, 6]]

    # A = [[3, 0, 0, 0, 0], 
    #     [0, -1, 0, 0, 0], 
    #     [0, 0, -1, 0, 0],
    #     [0, 0, 0, 0.5, 0.5], 
    #     [0, 0, 0, -0.5, 0.5]]

    # B = [[2, 0, 0, 0, 0], 
    #     [0, 1, 0, 0, 0], 
    #     [0, 0, 0.5, 0, 0],
    #     [0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 1]]

    # L, L_vec = sc.linalg.eig(A, B)
    # print ("lmda >> ")
    # print (L)

    # X0 = init_X0(len(A), p)

    # ################### 有限要素法ハンドブック
    # # Z_ = np.dot(B, X0)
    # # I = np.eye(p)
    # # print ("X0 >> ")
    # # print (X0)
    # # print ("Z_ >> ")
    # # print (Z_)

    # # while True:
    # #     X_new = np.dot( np.linalg.inv(A), Z_)
    # #     print ("X_new >> ", X_new)
    # #     Z_new = np.dot(B, X_new)
    # #     print ("Z_new >> ", Z_new)
    # #     # ここから変かも
    # #     X_new = np.dot(np.linalg.pinv(Z_new).T, I)
    # #     print ("Orthogonalization X_new >> ", X_new)
    # #     print ("tasikame >> ")
    # #     print (np.dot(X_new.T, Z_new))
        
    # #     Z_new = np.dot(A, X_new)
    # #     lmda, Q = Eig_matrix(A, X_new, p)
    # #     Z_ = np.dot(Z_new, Q)
    # #     print ("Q >> ")
    # #     print (Q)
    # #     print ("lambda >> ")
    # #     print (lmda)

    # #     a = input()
    # ################### 有限要素法ハンドブック


    # # Z_ = np.dot(B, X0)
    # I = np.eye(p)
    # print ("X0 >> ")
    # print (X0)
    # # print ("Z_ >> ")
    # # print (Z_)

    # while True:
    #     Z_ = np.dot(B, X0)

    #     Y = np.dot( np.linalg.inv(A), Z_)
    #     print ("X_new >> ", X0)

    #     A_new = np.dot(np.dot(Y.T, A), Y)
    #     B_new = np.dot(np.dot(Y.T, B), Y)
    #     print ("A_new >> ")
    #     print (A_new)
    #     print ("B_new >> ")
    #     print (B_new)

    #     lmda, Q = Eig_matrix_2(A_new, B_new)
    #     # eig_val,eig_vec =  sc.linalg.eig(A_new,B_new)
    #     # for i in range(len(eig_vec)): #正規化
    #         # eig_vec [i] = eig_vec[i]/np.linalg.norm(eig_vec[i])

    #     # print ("P >> ")
    #     # print (eig_vec)
    #     # print ("lambda >> ")
    #     # print (eig_val)

    #     X0 = np.dot(Y, Q)

    #     a = input()


