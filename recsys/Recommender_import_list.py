#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/19

@author: Maurizio Ferrari Dacrema
"""


######################################################################
##########                                                  ##########
##########                  NON PERSONALIZED                ##########
##########                                                  ##########
######################################################################
from recsys.Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects



######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
from recsys.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from recsys.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from recsys.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from recsys.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from recsys.GraphBased.P3alphaRecommender import P3alphaRecommender
from recsys.GraphBased.RP3betaRecommender import RP3betaRecommender
from recsys.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from recsys.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from recsys.MatrixFactorization.IALSRecommender import IALSRecommender
from recsys.MatrixFactorization.NMFRecommender import NMFRecommender
from recsys.EASE_R.EASE_R_Recommender import EASE_R_Recommender



######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
from recsys.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from recsys.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender



######################################################################
##########                                                  ##########
##########                       HYBRID                     ##########
##########                                                  ##########
######################################################################
from recsys.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from recsys.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender

