import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import shutil
import csv
from sympy import *
from pathlib import Path
from PIL import Image

d = Function("d",commutative=False)
inv = Function("inv",commutative=False)

def matrices(names):
    ''' Call with  A,B,C = matrix('A B C') '''
    return symbols(names,commutative=False)

class t(Function):
    ''' The transposition, with special rules
        t(A+B) = t(A) + t(B) and t(AB) = t(B)t(A) '''
    is_commutative = False
    def __new__(cls,arg):
        if arg.is_Add:
            return Add(*[t(A) for A in arg.args])
        elif arg.is_Mul:
            L = len(arg.args)
            return Mul(*[t(arg.args[L-i-1]) for i in range(L)])
        else:
            return Function.__new__(cls,arg)


MATRIX_DIFF_RULES = {
    # e =expression, s = a list of symbols respsect to which
    # we want to differentiate

    Symbol: lambda e, s: d(e) if (e in s) else 0,
    Add: lambda e, s: Add(*[matDiff(arg, s) for arg in e.args]),
    Mul: lambda e, s: Mul(matDiff(e.args[0], s), Mul(*e.args[1:]))
                      + Mul(e.args[0], matDiff(Mul(*e.args[1:]), s)),
    t: lambda e, s: t(matDiff(e.args[0], s)),
    inv: lambda e, s: - e * matDiff(e.args[0], s) * e
}

def matDiff(expr,symbols):
    if expr.__class__ in MATRIX_DIFF_RULES.keys():
        return  MATRIX_DIFF_RULES[expr.__class__](expr,symbols)
    else:
        return 0


class matStrPrinter(StrPrinter):
    ''' Nice printing for console mode : X¯¹, X', ∂X '''

    def _print_inv(self, expr):
        if expr.args[0].is_Symbol:
            return self._print(expr.args[0]) + '¯¹'
        else:
            return '(' + self._print(expr.args[0]) + ')¯¹'

    def _print_t(self, expr):
        return self._print(expr.args[0]) + "'"

    def _print_d(self, expr):
        if expr.args[0].is_Symbol:
            return '∂' + self._print(expr.args[0])
        else:
            return '∂(' + self._print(expr.args[0]) + ')'

Basic.__str__ = lambda self: matStrPrinter().doprint(self)

def matPrint(m):
    mem = Basic.__str__
    Basic.__str__ = lambda self: matStrPrinter().doprint(self)
    print(str(m).replace('*',''))
    Basic.__str__ = mem

if __name__ == '__main__':
    file = "/home/seonghun/Desktop/수업/2학기/Pattern Recognition/HW1/pattern_recog_ch3_data.csv"
    for i in range(3):
        for j in range(3):
            locals()["w{}x{}".format(i+1, j+1)] = []

    with open(file, "r") as f:
        reader = csv.reader(f)


        for idx, txt in enumerate(reader):
            if idx > 2:
                for i in range(3):
                    for j in range(3):
                        locals()["w{}x{}".format(i+1, j+1)] = np.append(locals()["w{}x{}".format(i+1, j+1)], np.float(txt[i*3 + j+1]))

    ### check local variable
    # for i in range(3):
    #     for j in range(3):
    #         print(locals()["w{}x{}".format(i+1, j+1)])


    print("Problem A")
    x, k, mu, sig = symbols('x k mu sig')
    diff_mu = diff(-log(2 * pi.evalf() * sig) / 2 - ((x - mu) ** 2) / (2 * sig), mu)
    diff_sig = diff(-log(2 * pi.evalf() * sig) / 2 - ((x - mu) ** 2) / (2 * sig), sig)

    print("mu, sigma derivate results")
    print(diff_mu)
    print(diff_sig)

    diff_mu = -(10 * mu - w1x1.sum())/sig
    diff_sig = -5/sig + (mu**2 - 2 * mu * w1x1.sum() + (w1x1**2).sum())/(2*sig**2)

    print()
    print("put value manually")
    print(diff_mu)
    print(diff_sig)


    mu = float(solve(Eq(diff_mu, 0), mu)[0])
    x_minus_mu = w1x1 - mu
    diff_sig = -5/sig + ((x_minus_mu**2).sum())/(2*(sig**2))


    print("solve equation and comparison for w1 x1")
    print("mean = {}".format(mu))
    print("real mean = {}".format(w1x1.mean()))
    print("sigma = {}".format(solve(Eq(diff_sig, 0), sig)[0]))
    print("real sigma = {}".format((w1x1**2).mean() - w1x1.mean()**2))

    print()
    print()
    x, k, mu, sig = symbols('x k mu sig')

    diff_mu = -(10 * mu - w1x2.sum())/sig
    diff_sig = -5/sig + (mu**2 - 2 * mu * w1x2.sum() + (w1x2**2).sum())/(2*sig**2)



    mu = float(solve(Eq(diff_mu, 0), mu)[0])
    x_minus_mu = w1x2 - mu
    diff_sig = -5/sig + ((x_minus_mu**2).sum())/(2*(sig**2))


    print("solve equation and comparison for w1 x2")
    print("mean = {}".format(mu))
    print("real mean = {}".format(w1x2.mean()))
    print("sigma = {}".format(solve(Eq(diff_sig, 0), sig)[0]))
    print("real sigma = {}".format((w1x2**2).mean() - w1x2.mean()**2))
    print()
    print()

    x, k, mu, sig = symbols('x k mu sig')
    diff_mu = -(10 * mu - w1x3.sum())/sig
    diff_sig = -5/sig + (mu**2 - 2 * mu * w1x3.sum() + (w1x3**2).sum())/(2*sig**2)


    mu = float(solve(Eq(diff_mu, 0), mu)[0])
    x_minus_mu = w1x3 - mu
    diff_sig = -5/sig + ((x_minus_mu**2).sum())/(2*(sig**2))

    print("solve equation and comparison for w1 x3")
    print("mean = {}".format(mu))
    print("real mean = {}".format(w1x3.mean()))
    print("sigma = {}".format(solve(Eq(diff_sig, 0), sig)[0]))
    print("real sigma = {}".format((w1x3**2).mean() - w1x3.mean()**2))

    print()
    print()


    print("Problem B, let we already know mean and covariance equation")
    print("case x1, x2")

    mean_mat = np.array([w1x1.mean(), w1x2.mean()]).T
    print("mean matrix [mu1, mu2] = ", end="")
    print(mean_mat)

    covar_mat = np.zeros((2,2))
    for idx in range(10):
        x_mu_sub_mat = np.array([[w1x1[idx], w1x2[idx]]]) - mean_mat.T
        covar_mat += np.matmul(x_mu_sub_mat.T, x_mu_sub_mat)
        # print(np.matmul(x_mu_sub_mat.T, x_mu_sub_mat))
    covar_mat /= 10
    print("covariance matrix = ", end="")
    print(covar_mat)

    print()
    print("case x1, x3")

    mean_mat = np.array([w1x1.mean(), w1x3.mean()]).T
    print("mean matrix [mu1, mu3] = ", end="")
    print(mean_mat)

    covar_mat = np.zeros((2,2))
    for idx in range(10):
        x_mu_sub_mat = np.array([[w1x1[idx], w1x3[idx]]]) - mean_mat.T
        covar_mat += np.matmul(x_mu_sub_mat.T, x_mu_sub_mat)
        # print(np.matmul(x_mu_sub_mat.T, x_mu_sub_mat))
    covar_mat /= 10
    print("covariance matrix = ", end="")
    print(covar_mat)

    print()
    print("case x3, x2")

    mean_mat = np.array([w1x3.mean(), w1x2.mean()]).T
    print("mean matrix [mu3, mu2] = ", end="")
    print(mean_mat)

    covar_mat = np.zeros((2,2))
    for idx in range(10):
        x_mu_sub_mat = np.array([[w1x3[idx], w1x2[idx]]]) - mean_mat.T
        covar_mat += np.matmul(x_mu_sub_mat.T, x_mu_sub_mat)
        # print(np.matmul(x_mu_sub_mat.T, x_mu_sub_mat))
    covar_mat /= 10
    print("covariance matrix = ", end="")
    print(covar_mat)

    print()
    print()

    print("Problem C")

    mean_mat = np.array([w1x1.mean(), w1x2.mean(), w1x3.mean()]).T
    print("mean matrix [mu1, mu2, mu3] = ", end="")
    print(mean_mat)

    covar_mat = np.zeros((3,3))
    for idx in range(10):
        x_mu_sub_mat = np.array([[w1x1[idx], w1x2[idx], w1x3[idx]]]) - mean_mat.T
        covar_mat += np.matmul(x_mu_sub_mat.T, x_mu_sub_mat)
        # print(np.matmul(x_mu_sub_mat.T, x_mu_sub_mat))
    covar_mat /= 10
    print("covariance matrix = ", end="")
    print(covar_mat)

    print()
    print("Problem D : same as Problem 1")

    x, y, z = symbols('x y z')
    mean_mat = np.array([w2x1.mean(), w2x2.mean(), w2x3.mean()])
    w2x1_minus_mean = w2x1 - mean_mat[0]
    w2x2_minus_mean = w2x2 - mean_mat[1]
    w2x3_minus_mean = w2x3 - mean_mat[2]
    sig1 = 1/10 * ((w2x1_minus_mean)**2).sum()
    sig2 = 1/10 * ((w2x2_minus_mean)**2).sum()
    sig3 = 1/10 * ((w2x3_minus_mean)**2).sum()
    covar_mat = [[sig1,0,0],
                 [0,sig2,0],
                 [0,0,sig3]]

    print(covar_mat)

    print()

    print("Problem E and F")
    x, k, mu, sig = symbols('x k mu sig')
    diff_mu = diff(-log(2 * pi.evalf() * sig) / 2 - ((x - mu) ** 2) / (2 * sig), mu)
    diff_sig = diff(-log(2 * pi.evalf() * sig) / 2 - ((x - mu) ** 2) / (2 * sig), sig)

    diff_mu = -(10 * mu - w2x1.sum())/sig
    diff_sig = -5/sig + (mu**2 - 2 * mu * w2x1.sum() + (w2x1**2).sum())/(2*sig**2)



    mu = float(solve(Eq(diff_mu, 0), mu)[0])
    x_minus_mu = w2x1 - mu
    diff_sig = -5/sig + ((x_minus_mu**2).sum())/(2*(sig**2))


    print("solve equation and comparison for w2 x1")
    print("mean = {}".format(mu))
    print("real mean = {}".format(w2x1.mean()))
    print("sigma = {}".format(solve(Eq(diff_sig, 0), sig)[0]))
    print("real sigma = {}".format((w2x1**2).mean() - w2x1.mean()**2))

    print()
    print()
    x, k, mu, sig = symbols('x k mu sig')

    diff_mu = -(10 * mu - w2x2.sum())/sig
    diff_sig = -5/sig + (mu**2 - 2 * mu * w2x2.sum() + (w2x2**2).sum())/(2*sig**2)



    mu = float(solve(Eq(diff_mu, 0), mu)[0])
    x_minus_mu = w2x2 - mu
    diff_sig = -5/sig + ((x_minus_mu**2).sum())/(2*(sig**2))


    print("solve equation and comparison for w2 x2")
    print("mean = {}".format(mu))
    print("real mean = {}".format(w2x2.mean()))
    print("sigma = {}".format(solve(Eq(diff_sig, 0), sig)[0]))
    print("real sigma = {}".format((w2x2**2).mean() - w2x2.mean()**2))
    print()
    print()

    x, k, mu, sig = symbols('x k mu sig')
    diff_mu = -(10 * mu - w2x3.sum())/sig
    diff_sig = -5/sig + (mu**2 - 2 * mu * w2x3.sum() + (w2x3**2).sum())/(2*sig**2)


    mu = float(solve(Eq(diff_mu, 0), mu)[0])
    x_minus_mu = w2x3 - mu
    diff_sig = -5/sig + ((x_minus_mu**2).sum())/(2*(sig**2))

    print("solve equation and comparison for w2 x3")
    print("mean = {}".format(mu))
    print("real mean = {}".format(w2x3.mean()))
    print("sigma = {}".format(solve(Eq(diff_sig, 0), sig)[0]))
    print("real sigma = {}".format((w2x3**2).mean() - w2x3.mean()**2))

    print()
    print()

    x, k, mu, sig = symbols('x k mu sig')
    diff_mu = diff(-log(2 * pi.evalf() * sig) / 2 - ((x - mu) ** 2) / (2 * sig), mu)
    diff_sig = diff(-log(2 * pi.evalf() * sig) / 2 - ((x - mu) ** 2) / (2 * sig), sig)


    diff_mu = -(10 * mu - w3x1.sum())/sig
    diff_sig = -5/sig + (mu**2 - 2 * mu * w3x1.sum() + (w3x1**2).sum())/(2*sig**2)



    mu = float(solve(Eq(diff_mu, 0), mu)[0])
    x_minus_mu = w3x1 - mu
    diff_sig = -5/sig + ((x_minus_mu**2).sum())/(2*(sig**2))


    print("solve equation and comparison for w3 x1")
    print("mean = {}".format(mu))
    print("real mean = {}".format(w3x1.mean()))
    print("sigma = {}".format(solve(Eq(diff_sig, 0), sig)[0]))
    print("real sigma = {}".format((w3x1**2).mean() - w3x1.mean()**2))

    print()
    print()
    x, k, mu, sig = symbols('x k mu sig')

    diff_mu = -(10 * mu - w3x2.sum())/sig
    diff_sig = -5/sig + (mu**2 - 2 * mu * w3x2.sum() + (w3x2**2).sum())/(2*sig**2)



    mu = float(solve(Eq(diff_mu, 0), mu)[0])
    x_minus_mu = w3x2 - mu
    diff_sig = -5/sig + ((x_minus_mu**2).sum())/(2*(sig**2))


    print("solve equation and comparison for w3 x2")
    print("mean = {}".format(mu))
    print("real mean = {}".format(w3x2.mean()))
    print("sigma = {}".format(solve(Eq(diff_sig, 0), sig)[0]))
    print("real sigma = {}".format((w3x2**2).mean() - w3x2.mean()**2))
    print()
    print()

    x, k, mu, sig = symbols('x k mu sig')
    diff_mu = -(10 * mu - w3x3.sum())/sig
    diff_sig = -5/sig + (mu**2 - 2 * mu * w3x3.sum() + (w3x3**2).sum())/(2*sig**2)


    mu = float(solve(Eq(diff_mu, 0), mu)[0])
    x_minus_mu = w3x3 - mu
    diff_sig = -5/sig + ((x_minus_mu**2).sum())/(2*(sig**2))

    print("solve equation and comparison for w3 x3")
    print("mean = {}".format(mu))
    print("real mean = {}".format(w3x3.mean()))
    print("sigma = {}".format(solve(Eq(diff_sig, 0), sig)[0]))
    print("real sigma = {}".format((w3x3**2).mean() - w3x3.mean()**2))

    print()
    print()