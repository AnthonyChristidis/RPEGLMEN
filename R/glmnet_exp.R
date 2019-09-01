
#' @title Elastic Net Penalized Exponentially Distributed Response Variables
#'
#' @description \code{git.glmGammaNet} Fit glmnet model for exponentiall distributed response data.
#'
#' @param A The matrix of independent variables.
#' @param b The vector of response variables.
#' @param alpha.EN The coefficient of elastic net regularizer (1 means lasso).
#' @param num_lambda Size of the lambda grid.
#' @param glm_type Type of glm model, 1 is exponential, 2 is gamma (not implemented yet).
#' @param max_iter Max number of iteration for the prox grad descent optimizer.
#' @param abs_tol Absolute error threshold for the pgd optimizer.
#' @param rel_tol Relative error threshold for the pgd optimizer (not used for vanilla PGD).
#' @param normalize_grad Swtich for whether to normalize the gradient or not.
#' @param k_fold The number of folds for cross validation.
#' @param k_fold_iter The number of iterations for the cross-validation.
#' @param has_intercept Parameter to determine if there is an intercept (TRUE) or not (FALSE).
#' @param ... Additional Parameters.
#'
#' @return Vector of optimal coefficient for the glm model.
#' 
#' @export
#' 
glmnet_exp <- function(A,
                      b,
                      alpha.EN = 0.5,
                      num_lambda = 100L,
                      glm_type = 1L,
                      max_iter = 100L,
                      abs_tol = 1.0e-4,
                      rel_tol = 1.0e-2,
                      normalize_grad = FALSE,
                      k_fold = 5L,
                      has_intercept = TRUE,
                      k_fold_iter = 5L,
                      ...){
  
  # Testing input for independent variables
  if(!is.matrix(A))
    stop("A must be a matrix.")
  # Testing input for response variable
  if(!is.vector(b))
    stop("b must be a vector of responses.")
  
  return(fitGlmCv(A,
         b,
         alpha = alpha.EN,
         num_lambda = num_lambda,
         glm_type = glm_type,
         max_iter = max_iter,
         abs_tol = abs_tol,
         rel_tol = rel_tol,
         normalize_grad = normalize_grad,
         k_fold = k_fold,
         has_intercept = has_intercept,
         k_fold_iter = k_fold_iter))
}
