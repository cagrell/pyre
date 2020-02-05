#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import math
import scipy.optimize as opt
import scipy.special as spec

from .distribution import *
from .normal import *

class Truncated_Normal(Distribution):
    """Truncated Normal distribution

    :Attributes:
    - name (str):         Name of the random variable\n
    - mean (float):       Mean\n
    - stdv (float):       Standard deviation\n
    - lb (float):         Lower bound
    - ub (float):         Upper bound
    """

    def __init__(self, name, mean, stdv, lb, ub, input_type=None,startpoint=None):
        self.type = 17
        self.distribution = {17:'Truncated_Normal'}
        self.mean = mean
        self.stdv = stdv
        self.lb = lb
        self.ub = ub
        a, b = (lb - mean) / stdv, (ub - mean) / stdv
        self.a = a
        self.b = b
        self.p1 = a
        self.p2 = b
        mean,stdv,p1,p2,p3,p4 = self.setMarginalDistribution()
        Distribution.__init__(self,name,self.type,mean,stdv,startpoint,p1,p2,p3,p4,input_type)


    def setMarginalDistribution(self):
        """Compute the marginal distribution  
        """
        return self.mean, self.stdv, self.a, self.b, 0, 0


    @classmethod
    def pdf(self,x,mean=None,stdv=None,a=None,b=None):
        """probability density function
        """
        p = sp.stats.truncnorm.pdf(x = x, a = a, b = b, loc = mean, scale = stdv)
        return p


    @classmethod
    def cdf(self,x,mean=None,stdv=None,a=None,b=None):
        """cumulative distribution function
        """
        P = sp.stats.truncnorm.cdf(x = x, a = a, b = b, loc = mean, scale = stdv)
        return P


    @classmethod
    def inv_cdf(self,P,mean=None,stdv=None,a=None,b=None):
        """inverse cumulative distribution function
        """
        x = sp.stats.truncnorm.ppf(q = P, a = a, b = b, loc = mean, scale = stdv)
        return x


    @classmethod
    def u_to_x(self, u, marg, x=None):
        """Transformation from u to x
        """
        if x == None:
            x = np.zeros(len(u))
        for i in range(len(u)):
            x[i] = Truncated_Normal.inv_cdf( Normal.cdf(u[i], 0, 1), marg.getMean(), marg.getStdv(), marg.getP1(), marg.getP2() )
        return x


    @classmethod
    def x_to_u(self, x, marg, u=None):
        """Transformation from x to u
        """
        if u == None:
            u = np.zeros(len(x))
        for i in range(len(x)):
            u[i] = Normal.inv_cdf( Truncated_Normal.cdf(x[i],marg.getMean(),marg.getStdv(), marg.getP1(), marg.getP2()) )
        return u

    @classmethod
    def jacobian(self,u,x,marg,J=None):
        """Compute the Jacobian
        """
        if J == None:
            J = np.zeros((len(marg),len(marg)))
        for i in range(len(marg)):
            pdf1 = Truncated_Normal.pdf(x[i],marg.getMean(),marg.getStdv(), marg.getP1(), marg.getP2())
            pdf2 = Normal.pdf(u[i],0,1)
            J[i][i] = pdf1*(pdf2)**(-1)
        return J