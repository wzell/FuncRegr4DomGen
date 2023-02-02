# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:09:38 2023

@author: ...
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.stats import truncnorm
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF

np.random.seed(0)

space = np.linspace(0, 1, 100)
np.random.shuffle(space)
color = cm.prism(space)

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def oscillating_parabola(nr_src_domains = 30,
                         nr_trgt_domains = 4,
                         sample_size = 200,
                         test_split = 0.3):
    N = nr_src_domains+nr_trgt_domains
    n_tr_examples = int(np.round(sample_size*(1-test_split)))
    domain_means = np.random.uniform(.3,.7,N)
    domain_latents = np.random.uniform(2,10,N)
    y_noise = np.random.normal(0,0.02,(sample_size, N))
    X = get_truncated_normal(mean=0, sd=1/8, low=-.3, upp=.3)
    x = X.rvs((sample_size, N))
    for i in range(N):
        x[:,i]*=domain_latents[i]/10
        x[:,i]+=domain_means[i]
    y = np.sin(x*300/(domain_means*10)**2)/10+.9-((x-0.5)*1.7)**2+y_noise
    x_src_train = x[:n_tr_examples,:nr_src_domains]
    x_src_test = x[n_tr_examples:,:nr_src_domains]
    x_trgt_train = x[:n_tr_examples,nr_trgt_domains:]
    x_trgt_test = x[n_tr_examples:,nr_trgt_domains:]
    y_src_train = y[:n_tr_examples,:nr_src_domains]
    y_src_test = y[n_tr_examples:,:nr_src_domains]
    y_trgt_train = y[:n_tr_examples,nr_trgt_domains:]
    y_trgt_test = y[n_tr_examples:,nr_trgt_domains:]
    return x_src_train, x_src_test, y_src_train, y_src_test, \
           x_trgt_train, x_trgt_test, y_trgt_train, y_trgt_test
           
def calc_mean_embeddings(x, mesh=np.arange(0,1,1/1000), kernel=ExpSineSquared(periodicity=1,length_scale=1)):
    # takes x of the form [n_examples, n_domains]
    X_embedded = np.zeros((mesh.shape[0],x.shape[1]))# n_samples x n_domains
    for domain in range(x.shape[1]):
        mean_embedding=pairwise_kernels(x[:,domain].reshape(-1,1),
                                        mesh.reshape(-1,1),
                                        ExpSineSquared(periodicity=1)).mean(axis=0)
        X_embedded[:,domain]=mean_embedding  
    return X_embedded

def compute_regressors(x, y, mesh=np.arange(0,1,1/1000)):
    # Kernel ridge regression with GridCV for all domains
    kernel_ridge = KernelRidge(kernel=ExpSineSquared(),alpha=1e-3)
    params = {
        "alpha": [1e-1,1e-2,1e-3,1e-4],
        "kernel__length_scale": [1e-2,1e-1,1],
        "kernel__periodicity": [1,2,3,5,10]
    }
    Y_pred = np.zeros((mesh.shape[0],x.shape[1]))
    for domain in range(x.shape[1]):
        kernel_ridge_tuned = GridSearchCV(kernel_ridge, param_grid=params)
        kernel_ridge_tuned.fit(x[:,domain].reshape(-1,1), y[:,domain])
        Y_pred[:,domain] = kernel_ridge_tuned.predict(mesh.reshape(-1,1))
    return Y_pred

def mean_centering(X):
    x_mean = X.mean(axis=1)
    for domain in range(X.shape[1]): X[:,domain]-=x_mean
    return X, x_mean
    
def slope_intercept(X, Y, x_mean, y_mean, kernel=ExpSineSquared(),
                    mesh=np.arange(0,1,1/1000), lambda_ridge = 1e-7):
    # K kernel matrix between grid points
    K_kernel_mat=pairwise_kernels(mesh.reshape(-1,1), mesh.reshape(-1,1), metric=kernel)
    # G matrix
    G = X.transpose() @ K_kernel_mat @ X
    # Slope and intercept regression
    B = inv(G+X.shape[1]*lambda_ridge*np.eye(X.shape[1])) @ Y.transpose()
    X_slopes = X.transpose() @ K_kernel_mat
    slope = B.transpose() @ X_slopes
    intercept = y_mean.reshape(-1,1) - (slope @ x_mean.reshape(-1,1))
    return slope, intercept

def predict_functions(x, slope, intercept, mesh=np.arange(0,1,1/1000),
                      kernel=ExpSineSquared(periodicity=1,length_scale=1)):
    X_embedded = calc_mean_embeddings(x, mesh, kernel)
    X_embedded, x_mean = mean_centering(X_embedded)
    x_for_pred = X_embedded + x_mean.reshape(-1,1)
    return intercept.reshape(-1,1) + (slope @ x_for_pred)

def predict_nearest_neighbor(x, y_mesh, mesh=np.arange(0,1,1/1000)):
    y_pred = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        difference_array = np.absolute(mesh-x[i])
        mesh_ind = difference_array.argmin()
        y_pred[i] = y_mesh[mesh_ind]
    return y_pred

def squared_error(x_train, x_test, y_test, slope, intercept):
    y_pred_mesh = predict_functions(x_train.reshape(-1,1), slope, intercept)
    y_pred = predict_nearest_neighbor(x_test.reshape(-1,1), y_pred_mesh.reshape(-1,1))
    error = np.mean((y_pred.reshape(-1,1)-y_test.reshape(-1,1))**2,axis=0)
    return error

def cross_val_fold(domains_train, domains_val,
                   x_src_train_cv, y_src_train_cv,
                   x_trgt_train_cv, y_trgt_train_cv,
                   x_src_test_cv, y_src_test_cv,
                   x_trgt_test_cv, y_trgt_test_cv,
                   kernel_embedding = ExpSineSquared(periodicity=1,length_scale=1),
                   kernel_slope = ExpSineSquared(),
                   lambda_ridge = 1e-7,
                   xmesh = np.arange(0,1,1/1000)):
    
    # Our algorithm computed in a single cross-validation fold
    # steps 2 and 3 follow Algorithm 1 in: Tong, Hong Zhi, Ling Fang Hu, and Michael Ng.
    # "Non-asymptotic Error Bound for Optimal Prediction of Function-on-Function
    # Regression by RKHS Approach." Acta Mathematica Sinica, English Series 38.4 (2022): 777-796.
    
    # 1. Mean embedding
    X_src_train_cv_embedded = calc_mean_embeddings(x_src_train_cv, xmesh, kernel=kernel_embedding)
    # 2. Regression functions
    Y_src_train_cv_pred = compute_regressors(x_src_train_cv, y_src_train_cv, mesh=xmesh)
    # 3. Mean centering data
    X_src_train_cv_embedded, x_mean = mean_centering(X_src_train_cv_embedded)
    Y_src_train_cv_pred, y_mean = mean_centering(Y_src_train_cv_pred)
    # 4. Calculate slope and intercept
    slope, intercept = slope_intercept(X_src_train_cv_embedded, Y_src_train_cv_pred,
                                       x_mean, y_mean, lambda_ridge = lambda_ridge,
                                       kernel = kernel_slope)
    # Compute errors (src_train, src_test, target_train, target_test)
    src_errors = np.zeros((2,domains_train.shape[0]))
    val_errors = np.zeros((2,domains_val.shape[0]))
    for domain in range(domains_train.shape[0]):
        src_errors[0,domain] = squared_error(x_src_train_cv[:,domain],
                                             x_src_train_cv[:,domain],
                                             y_src_train_cv[:,domain],
                                             slope, intercept)
        src_errors[1,domain] = squared_error(x_src_train_cv[:,domain],
                                             x_src_test_cv[:,domain],
                                             y_src_test_cv[:,domain],
                                             slope, intercept)
    for domain in range(domains_val.shape[0]):
        val_errors[0,domain] = squared_error(x_trgt_train_cv[:,domain],
                                             x_trgt_train_cv[:,domain],
                                             y_trgt_train_cv[:,domain],
                                             slope, intercept)
        val_errors[1,domain] = squared_error(x_trgt_train_cv[:,domain],
                                             x_trgt_test_cv[:,domain],
                                             y_trgt_test_cv[:,domain],
                                             slope, intercept)
    four_errors = np.zeros((2,2))
    four_errors[:,0]=src_errors.mean(axis=1)
    four_errors[:,1]=val_errors.mean(axis=1)
    return four_errors

def split_and_fold(domains_train, domains_val, x_src_train, y_src_train,
                   x_src_test, y_src_test,
                   kernel_embedding = ExpSineSquared(periodicity=1,length_scale=1),
                   kernel_slope = ExpSineSquared(),
                   lambda_ridge = 1e-7,
                   xmesh = np.arange(0,1,1/1000)):
    # Split data
    x_src_train_cv = x_src_train[:,domains_train]
    y_src_train_cv = y_src_train[:,domains_train]
    x_trgt_train_cv = x_src_train[:,domains_val]
    y_trgt_train_cv = y_src_train[:,domains_val]
    x_src_test_cv = x_src_test[:,domains_train]
    y_src_test_cv = y_src_test[:,domains_train]
    x_trgt_test_cv = x_src_test[:,domains_val]
    y_trgt_test_cv = y_src_test[:,domains_val]
    
    four_errors = cross_val_fold(domains_train, domains_val,
                                 x_src_train_cv, y_src_train_cv,
                                 x_trgt_train_cv, y_trgt_train_cv,
                                 x_src_test_cv, y_src_test_cv,
                                 x_trgt_test_cv, y_trgt_test_cv,
                                 kernel_embedding = kernel_embedding,
                                 kernel_slope=kernel_slope,
                                 lambda_ridge = lambda_ridge,
                                 xmesh = xmesh)
    return four_errors

def cv(nr_domains_cv,
       xmesh = np.arange(0,1,1/1000),
       val_split = 0.2,
       kernel_embedding = ExpSineSquared(periodicity=1,length_scale=1),
       kernel_slope = ExpSineSquared(periodicity=1,length_scale=1),
       lambda_ridge = 1e-3):
    # Cross-validation (only 5-fold CV tested)
    nr_vals = int(np.round(nr_domains_cv*val_split))
    errors = np.zeros((5,2,2))
    fold_nr = 0
    for ind_start in np.arange(5)*nr_vals:
        # Data split indices
        domains_val = np.arange(ind_start,ind_start+nr_vals)
        domains_train = np.concatenate([np.arange(ind_start),
                                        np.arange(ind_start+nr_vals,nr_domains_cv)])
        
        errors[fold_nr] = split_and_fold(domains_train, domains_val,
                                         x_src_train, y_src_train,
                                         x_src_test, y_src_test,
                                         kernel_embedding = kernel_embedding,
                                         kernel_slope = kernel_slope,
                                         lambda_ridge = lambda_ridge,
                                         xmesh = xmesh)
        # print('Fold='+str(fold_nr))
        # print(errors[fold_nr])
        fold_nr+=1
    # print('CV done...')
    return errors

# -----------------------------------------------------------------------------
# Generate data
x_src_train, x_src_test, y_src_train, y_src_test, \
x_trgt_train, x_trgt_test, y_trgt_train, y_trgt_test \
= oscillating_parabola(nr_src_domains = 100,
                       nr_trgt_domains = 20,
                       sample_size = 200,
                       test_split = 0.3)
# -----------------------------------------------------------------------------
# # Meta cross-validation
# xmesh = np.arange(0,1,1/1000)
# nr_domains_cv = x_src_train.shape[1]
# val_split = 0.2 # 5-fold cross-validation
# kernels_embedding = [RBF(length_scale=1e-2),
#                      RBF(length_scale=1e-3),
#                      ExpSineSquared(periodicity=1,length_scale=1)]
                     
# kernels_slope = [RBF(length_scale=1e-2),
#                  RBF(length_scale=1e-3),
#                  ExpSineSquared(periodicity=1,length_scale=1),
#                  ExpSineSquared(periodicity=1,length_scale=1e-2),
#                  ExpSineSquared(periodicity=1,length_scale=1e-3),
#                  RBF(length_scale=1e-1)]         
# lambdas_ridge = [1e-3,1e-4,1e-5,1e-6]

# for kernel_embedding in kernels_embedding:
#     for kernel_slope in kernels_slope:
#         for lambda_ridge in lambdas_ridge:
#             print('Kernel_embedding='+str(kernel_embedding)+'\n'+
#                   'Kernel_slope='+str(kernel_slope)+'\n'+
#                   'Lambda_ridge='+str(lambda_ridge))
#             errors = cv(nr_domains_cv,
#                         xmesh = xmesh,
#                         val_split = val_split,
#                         kernel_embedding = kernel_embedding,
#                         kernel_slope = kernel_slope,
#                         lambda_ridge = lambda_ridge)
#             print(errors.mean(axis=0))
# # -----------------------------------------------------------------------------
# Testing with best parameters
# Our cross-validation result for the best parameters is:
# Kernel_embedding=RBF(length_scale=0.01)
# Kernel_slope=RBF(length_scale=0.01)
# Lambda_ridge=0.0001
xmesh = np.arange(0,1,1/1000)

# Mean embedding
X_src_train_cv_embedded = calc_mean_embeddings(x_src_train, xmesh, kernel=RBF(length_scale=0.01))
# Regression functions
Y_src_train_cv_pred = compute_regressors(x_src_train, y_src_train, mesh=xmesh)
# Mean centering data
X_src_train_cv_embedded, x_mean = mean_centering(X_src_train_cv_embedded)
Y_src_train_cv_pred, y_mean = mean_centering(Y_src_train_cv_pred)
# Calculate slope and intercept
slope, intercept = slope_intercept(X_src_train_cv_embedded, Y_src_train_cv_pred,
                                    x_mean, y_mean, lambda_ridge = 0.0001,
                                    kernel = RBF(length_scale=0.01))
# Compute errors (src_train, src_test, target_train, target_test)
src_errors = np.zeros((2,100))
val_errors = np.zeros((2,20))
for domain in range(100):
    src_errors[0,domain] = squared_error(x_src_train[:,domain],
                                          x_src_train[:,domain],
                                          y_src_train[:,domain],
                                          slope, intercept)
    src_errors[1,domain] = squared_error(x_src_train[:,domain],
                                          x_src_test[:,domain],
                                          y_src_test[:,domain],
                                          slope, intercept)
for domain in range(20):
    val_errors[0,domain] = squared_error(x_trgt_train[:,domain],
                                          x_trgt_train[:,domain],
                                          y_trgt_train[:,domain],
                                          slope, intercept)
    val_errors[1,domain] = squared_error(x_trgt_train[:,domain],
                                          x_trgt_test[:,domain],
                                          y_trgt_test[:,domain],
                                          slope, intercept)
four_errors = np.zeros((2,2))
four_errors[:,0]=src_errors.mean(axis=1)
four_errors[:,1]=val_errors.mean(axis=1)
# -----------------------------------------------------------------------------
# # Plotting
# plt.figure()
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# # plt.plot(xmesh,.9-((xmesh-0.5)*1.7)**2,'k-',lw=0.5)
# for i in range(4):
#     y_pred_mesh = predict_functions(x_trgt_train[:,i].reshape(-1,1), slope, intercept)
#     plt.plot(x_trgt_train[:200,i], y_trgt_train[:200,i], '.', alpha=0.3,color=color[i])
#     # y_trgt_test_pred = (np.sin(xmesh*300/(x_trgt_train[:,i].mean()*10)**2)/10+.9-((xmesh-0.5)*1.7)**2)
#     xmin = x_trgt_train[:,i].min()
#     xmax = x_trgt_train[:,i].max()
#     indizes = ((xmesh>=xmin) & (xmesh<=xmax))
#     plt.plot(xmesh[indizes], y_pred_mesh[indizes], '--', color=color[i])
# plt.savefig('pred_train.jpg', dpi=300)

plt.figure()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(xmesh,.9-((xmesh-0.5)*1.7)**2,'k-',lw=0.5)
for i in range(4):
    y_pred_mesh = predict_functions(x_trgt_test[:,i].reshape(-1,1), slope, intercept)
    plt.plot(x_trgt_test[:,i], y_trgt_test[:,i], '.', alpha=0.3,color=color[i])
    # y_trgt_test_pred = (np.sin(xmesh*300/(x_trgt_train[:,i].mean()*10)**2)/10+.9-((xmesh-0.5)*1.7)**2)
    plt.plot(xmesh, y_pred_mesh, '--', color=color[i])
plt.savefig('pred_test.jpg', dpi=300)































# # -----------------------------------------------------------------------------
# # X: Calculate mean embeddings of source inputs
# xmesh = np.arange(0,1,1/1000)
# X_src_train_embedded = calc_mean_embeddings(x_src_train, xmesh, ExpSineSquared(periodicity=1,length_scale=1))
# # -----------------------------------------------------------------------------
# # Y: Compute regressors on source domains
# Y_src_train_pred = compute_regressors(x_src_train, y_src_train, mesh=xmesh)
# # -----------------------------------------------------------------------------
# # Plotting
# plt.figure()
# for i in range(4):
#     plt.hist(x_src_train[:,i], 50, density=True, alpha=0.3, range=[0, 1], color=color[i])
# plt.figure()
# plt.plot(xmesh,.9-((xmesh-0.5)*1.7)**2,'k-',lw=0.5)
# for i in range(4):
#     plt.plot(x_src_train[:,i], y_src_train[:,i], '.', alpha=0.3,color=color[i])
#     plt.plot(xmesh, Y_src_train_pred[:,i], linewidth=1, linestyle="-", color=color[i])
# # -----------------------------------------------------------------------------
# # Mean centering of data X and Y
# X_src_train_embedded, x_mean = mean_centering(X_src_train_embedded)
# Y_src_train_pred, y_mean = mean_centering(Y_src_train_pred)
# # -----------------------------------------------------------------------------
# # Calculate slope and intercept
# slope, intercept = slope_intercept(X_src_train_embedded, Y_src_train_pred, x_mean, y_mean)
# # -----------------------------------------------------------------------------
# # Plot predictions for source train
# plt.figure()
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# y_pred = predict_functions(x_src_train, slope, intercept)
# for src_domain_nr in range(x_src_train.shape[1]):
#     # x_for_pred = X_src_train_embedded[:,src_domain_nr]+x_mean
#     # y_pred = predict_functions(x_for_pred, slope, intercept)
#     if src_domain_nr in range(4):
#         plt.plot(xmesh.reshape(-1,1), y_pred[:,src_domain_nr].reshape(-1,1),
#                  linestyle='-',color=color[src_domain_nr])
#         plt.plot(x_src_train[:,src_domain_nr], y_src_train[:,src_domain_nr], '.',
#                  alpha=0.3,color=color[src_domain_nr])
#         plt.plot(xmesh, Y_src_train_pred[:,src_domain_nr]+y_mean,
#                  linewidth=1, linestyle="--",color=color[src_domain_nr])
# # -----------------------------------------------------------------------------
# # Plot predictions for target train
# plt.figure()
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# y_pred = predict_functions(x_trgt_train, slope, intercept)
# for trgt_domain_nr in range(x_trgt_train.shape[1]):
#     if trgt_domain_nr in range(4):
#         plt.plot(xmesh.reshape(-1,1), y_pred[:,trgt_domain_nr].reshape(-1,1),
#                   linestyle='-',color=color[trgt_domain_nr])
#         plt.plot(x_trgt_train[:,trgt_domain_nr], y_trgt_train[:,trgt_domain_nr], '.',
#                   alpha=0.3,color=color[trgt_domain_nr])
# # -----------------------------------------------------------------------------
# # Plot predictions for target test
# plt.figure()
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# y_pred_mesh = predict_functions(x_trgt_train, slope, intercept)
# for domain in range(x_trgt_test.shape[1]):
#     y_pred = predict_nearest_neighbor(x_trgt_test[:,domain], y_pred_mesh[:,domain])
#     if domain in range(4):
#         plt.plot(x_trgt_test[:,domain], y_pred.reshape(-1,1),'x',color=color[domain])
#         plt.plot(x_trgt_test[:,domain], y_trgt_test[:,domain], '.',
#                  alpha=0.3,color=color[domain])
#         plt.plot(xmesh.reshape(-1,1), y_pred_mesh[:,domain].reshape(-1,1),
#                  linestyle='--',color=color[domain])
# # -----------------------------------------------------------------------------
# # Compute test error
# domain = 0
# y_pred_mesh = predict_functions(x_trgt_train[:,domain].reshape(-1,1), slope, intercept)
# y_pred = predict_nearest_neighbor(x_trgt_test[:,domain].reshape(-1,1), y_pred_mesh.reshape(-1,1))
# error = np.mean((y_pred-y_trgt_test[:,domain])**2,axis=0)
# # -----------------------------------------------------------------------------

