import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import sys
import time
import csv
import SubMatrix as AS
import scipy as sc

##############################################
##############################################
### 
### メッシュデータから入力値の読み込み
###     >> READ_materialDATA()
### Dマトリクス作成
###     >> D_matrix_make()
### Bマトリクス作成
###     >> B_matrix_make()
### 要素剛性マトリクス作成
###     >> Element_stiffness_matrix()
### 要素応力計算
###     >> Element_stress_calculate()
### 全体剛性マトリクス
###     >> K_matrix_make()
###
##############################################
##############################################

def READ_materialDATA():
    ##############################################
    ### 
    ### num_node: 節点数
    ### num_element: 要素数
    ### plane_status: 二次元弾性体の状態（0: 平面ひずみ，1: 平面応力）
    ### node: 節点情報（節点番号，x座標[m]，y座標[m]）
    ### element: 要素情報（要素番号，節点１，節点２，節点３）
    ### 
    ### 
    ##############################################

    input_dict = {}

    num_node = 4
    num_element = 2
    plane_status = 1
    # E = 1e+11
    # nu = 0.3
    E = 1875
    nu = 0.25
    node = [[1, 0, 0], [2, 0.1, 0], [3, 0.1, 0.1], [4, 0, 0.1]]
    element = [[1, 1, 2, 4], [2, 2, 3, 4]]
    F = np.array([0, -1, 0, -1])
    t = 1


    input_dict["num_node"] = num_node
    input_dict["num_element"] = num_element
    input_dict["plane_status"] = plane_status
    input_dict["E"] = E
    input_dict["nu"] = nu
    input_dict["node"] = node
    input_dict["element"] = element
    input_dict["F"] = F
    input_dict["T"] = t
    
    return input_dict

def Input_Data():
    ##############################################
    ### ヒューマンインプットデータ
    ### 
    ### 固有周波数値の取得数
    ### npfix: 拘束条件 (1: 拘束, 0: 自由)
    ### 
    ##############################################
    input_dict = {}
    
    npfix = [[1, 1, 1], 
        [2, 1, 1], 
        [3, 0, 0], 
        [4, 0, 0]]

    input_dict["npfix"] = npfix

    return input_dict 

def D_matrix_make(plane_status, E, nu): # Dマトリクス作成
    d = np.zeros((3,3))
    if plane_status == 0: # plane strain
        d[0][0] = d[1][1] = 1 - nu
        d[0][1] = d[1][0] = nu
        d[2][2] = (1 - 2*nu)/2
        d = (E/((1 + nu)*(1 - 2*nu)))*d

    elif plane_status == 1: # plane stress
        d[0][0] = d[1][1] = 1
        d[0][1] = d[1][0] = nu
        d[2][2] = (1 - nu)/2
        d = (E/(1 - nu**2))*d
    
    return d

def B_matrix_make(node, element): # Bマトリクス作成
    print (node, element)
    x1 = node[element[1] - 1][1]
    y1 = node[element[1] - 1][2]
    x2 = node[element[2] - 1][1]
    y2 = node[element[2] - 1][2]
    x3 = node[element[3] - 1][1]
    y3 = node[element[3] - 1][2]

    B = np.zeros((3,6))
    area = ((x3 - x2)*y1 + (x1 - x3)*y2 + (x2 - x1)*y3)/2
    B[0][0] = y2 - y3; B[0][2] = y3 - y1; B[0][4] = y1 - y2;
    B[1][1] = x3 - x2; B[1][3] = x1 - x3; B[1][5] = x2 - x1;
    B[2][0] = x3 - x2; B[2][1] = y2 - y3; B[2][2] = x1 - x3;
    B[2][3] = y3 - y1; B[2][4] = x2 - x1; B[2][5] = y1 - y2;
    print ("B >> ")
    print (B)

    B = B/(2*area)
    return B, area


def Element_stiffness_matrix(plane_status, 
        t, 
        E, nu, 
        node, element):
        # 要素剛性マトリクス作成
    
    D = D_matrix_make(plane_status, E, nu)
    print ("D >> ")
    print (D)
    B, area = B_matrix_make(node, element)
    Es = np.dot(B.T, np.dot(D, B))*t*area

    # stress = Element_stress_calculate(plane_status, 
    #     E, nu, 
    #     alpha, tem, wd, 
    #     D, B, area)

    return Es

def Element_stress_calculate(plane_status,
        E, nu, 
        alpha, tem, wd, 
        D, B, area):
        # 要素応力計算
    eps0 = np.zeros(3)
    eps = np.dot(B, wd)

    if plane_status == 0:
        eps0[0] = eps0[1] = tem*(1 + nu)*alpha
        eps0[2] = 0.0

    elif plane_status == 1:
        eps0[0] = eps0[1] = tem*alpha
        eps0[2] = 0.0

    stress = np.dot(D, (eps - eps0))
    return stress

def K_matrix_make(plane_status, t, E, nu, 
        node, element):
    # 全体剛性マトリクス
    sp_idx = list()
    sp_element = list()
    K = np.zeros((2*len(node), 2*len(node)))

    for k in range(len(element)):
        Es = Element_stiffness_matrix(plane_status, t, E, nu, 
            node, element[k][:])

        for _p1 in range(3):
            for _p2 in range(3):
                for _q1 in range(2):
                    for _q2 in range(2):
                        if (Es[(_p1*2)+_q1][(_p2*2)+_q2] != 0.0):
                            # sp_element.append(Es[(_p1*2)+_q1][(_p2*2)+_q2])

                            # sp_idx.append([int(element[k][_p1 + 1] - 1)*2+_q1, int(element[k][_p2 + 1] - 1)*2+_q2])
                            ### Kフルマトリクス作成
                            K[int(element[k][_p1 + 1] - 1)*2+_q1][int(element[k][_p2 + 1] - 1)*2+_q2] += Es[(_p1*2)+_q1][(_p2*2)+_q2]

    return K

def Restriction_matrix(A, npfix, matrix_type = "X"):
    if matrix_type == "X":
        print ("ERROR: noselect matrix type")
    elif matrix_type == "K":
        K_new = np.zeros((len(A), len(A)))
        for 

def N_func(solid_type, L1_L2_L3):
    # 内挿関数　作成
    
    if solid_type == "N_tria3":
        N = np.zeros((2,2*3))
        L1 = L1_L2_L3[0]
        L2 = L1_L2_L3[1]
        L3 = L1_L2_L3[2]

        N[0][0] = N[1][1] = L1
        N[0][2] = N[1][3] = L2
        N[0][4] = N[1][5] = L3

    return N

def Jacobi_cal(solid_type, node, element):
    print ("node >> ")
    print (node)
    print ("element >> ")
    print (element)
    if solid_type == "N_tria3":
        dNd_L = np.zeros((3, 3))
        xy = np.zeros((3, 2))
        const_m = np.zeros((2, 3))

        const_m[0][0] = const_m[1][1] = 1
        const_m[0][2] = const_m[1][2] = -1

        dNd_L[0][0] = dNd_L[1][1] = dNd_L[2][2] = 1

        xy[0][0] = node[element[1] - 1][1]
        xy[0][1] = node[element[1] - 1][2]
        xy[1][0] = node[element[2] - 1][1]
        xy[1][1] = node[element[2] - 1][2]
        xy[2][0] = node[element[3] - 1][1]
        xy[2][1] = node[element[3] - 1][2]

        J = np.dot( np.dot(const_m, dNd_L), xy)
        # print ("cnt_ >> ")
        # print (np.dot(const_m, dNd_L))
        # print ("xy_ >> ")
        # print (xy)
        # print ("J >> ")
        # print (J)
    return np.linalg.det(J)

def Element_mass_matrix(solid_type, node, element, t):
    # print ("num_element >> ")
    # print (len(element))

        
    if solid_type == "N_tria3":

        L1_L2_L3 = np.zeros((3,3))

        L1_L2_L3[0][0] = L1_L2_L3[0][1] = 0.5
        L1_L2_L3[1][1] = L1_L2_L3[1][2] = 0.5
        L1_L2_L3[2][0] = L1_L2_L3[2][2] = 0.5

        Na = N_func("N_tria3", L1_L2_L3[0][:])
        Nb = N_func("N_tria3", L1_L2_L3[1][:])
        Nc = N_func("N_tria3", L1_L2_L3[2][:])

        det_J = Jacobi_cal("N_tria3", node, element)

        # print ("det_J >> ")
        # print (det_J)
        
        Ma = np.dot(Na.T, Na)*det_J*t
        Mb = np.dot(Nb.T, Nb)*det_J*t
        Mc = np.dot(Nc.T, Nc)*det_J*t

        m = (1/6)*Ma + (1/6)*Mb + (1/6)*Mc

    return m

def M_matrix_make(solid_type ,node, element,t):
    # # 質量全体マトリクス　作成
    M = np.zeros((2*len(node), 2*len(node)))
    for k in range(len(element)):
        m = Element_mass_matrix(solid_type, node, element[k][:], t)
        
        for _p1 in range(3):
            for _p2 in range(3):
                for _q1 in range(2):
                    for _q2 in range(2):
                        if (m[(_p1*2)+_q1][(_p2*2)+_q2] != 0.0):

                            M[int(element[k][_p1 + 1] - 1)*2+_q1][int(element[k][_p2 + 1] - 1)*2+_q2] += m[(_p1*2)+_q1][(_p2*2)+_q2]

    return M

def Restraint_Matrix():
    
    Z = np.zeros((len(B), len(B)))
    A_sum = np.sum(A[:, 1:3], axis = 1)

    print (A_sum)

    for k in range(len(A_sum)):
        if (A_sum[k] == 0): # 拘束されていないところ
            node_u = (A[k][0] - 1)*uvw # 
            for uvw_u in range(uvw):
                if (A[k][uvw_u+1] == 0): # 拘束されていないところ_u
                    for uvw_v in range(uvw):
                        if (A[k][uvw_v+1] == 0): # 拘束されていないところ_v
                            Z[node_u + uvw_u][node_u + uvw_v] = B[node_u + uvw_u][node_u + uvw_v]

    print ("Z >> ")
    print (Z)
    pass

if __name__ == "__main__":
    input_param_dict = READ_materialDATA()
    input_data_dict = Input_Data()

    ### ### ### ### ### ###
    num_node = input_param_dict["num_node"]
    num_element = input_param_dict["num_element"]
    plane_status = input_param_dict["plane_status"]
    E = input_param_dict["E"]
    nu = input_param_dict["nu"]
    node = input_param_dict["node"]
    element = input_param_dict["element"]
    t = input_param_dict["T"]
    F = input_param_dict["F"]
    ### ### ### ### ### ###

    ### ### ### ### ### ###
    npfix = input_data_dict["npfix"]
    ### ### ### ### ### ###


    D = D_matrix_make(plane_status, E, nu)

    K = K_matrix_make(plane_status, t, E, nu, node, element)
    print (K)
    # K_bc = K[4:8, 4:8]

    K_mtx = csr_matrix(K_bc)
    print ("K_bc >> ")
    print (K_bc)
    U = spsolve(K_mtx, F)
    print ("U >> ")
    print (U)

    with open('K/K.csv', 'w') as f:
        writer = csv.writer(f, lineterminator = "\n")
        writer.writerows(K)

    M = M_matrix_make("N_tria3", node, element, t)
    print ("M >> ")
    print (M)

    with open('K/M.csv', 'w') as f:
        writer = csv.writer(f, lineterminator = "\n")
        writer.writerows(M)
    

    # a = input()
    L, L_vec = sc.linalg.eig(K, M)
    print ("lmda_A >> ")
    print (L)
    
    Lmda = AS.Subspace(K, M, 6)
    print ("lmda >> ")
    print (Lmda)
    print ("lmda.real >> ")
    print (Lmda.real)
