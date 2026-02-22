#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

/* .C entry points */
void C_cv_deg0_binloss();
void C_cv_deg0_mae();
void C_epanechnikov_kernel_with_stop();
void C_triangular_kernel_with_stop();
void C_tr_exponential_kernel_with_stop();
void C_kernel_eval();
void C_llm_1D_beta();
void C_llm_1D_beta_perms();
void C_predict_1D();
void C_wpredict_1D();
void C_llm_1D_fit_and_predict();
void C_mllm_1D_fit_and_predict();
void C_llm_1D_fit_and_predict_BB_CrI();
void C_llm_1D_fit_and_predict_global_BB_CrI();
void C_llm_1D_fit_and_predict_global_BB();
void C_llm_1D_fit_and_predict_global_BB_qCrI();
void C_loo_llm_1D();
void C_deg0_loo_llm_1D();
void C_cv_mae_1D();
void C_get_BB_Eyg();
void C_get_Eyg_CrI();
void C_get_Eygs();
void C_get_bws();
void C_get_bws_with_minK_a();
void C_columnwise_weighting();
void C_columnwise_eval();
void C_columnwise_TS_norm();
void C_matrix_wmeans();
void C_columnwise_wmean();
void C_columnwise_wmean_BB();
void C_columnwise_wmean_BB_qCrI();
void C_columnwise_wmean_BB_CrI_1();
void C_columnwise_wmean_BB_CrI_2();
void C_normalize_dist();
void C_pearson_wcor();
void C_pearson_wcor_BB_qCrI();

/* .Call entry points */
SEXP S_generate_dirichlet_weights();
SEXP S_get_BB_Eyg_external();
SEXP S_magelog();
SEXP S_mabilog();
SEXP S_mabilo_plus();
SEXP S_mabilo();

static const R_CMethodDef cMethods[] = {
  {"C_cv_deg0_binloss", (DL_FUNC) &C_cv_deg0_binloss, 14, NULL},
  {"C_cv_deg0_mae", (DL_FUNC) &C_cv_deg0_mae, 14, NULL},
  {"C_epanechnikov_kernel_with_stop", (DL_FUNC) &C_epanechnikov_kernel_with_stop, 5, NULL},
  {"C_triangular_kernel_with_stop", (DL_FUNC) &C_triangular_kernel_with_stop, 5, NULL},
  {"C_tr_exponential_kernel_with_stop", (DL_FUNC) &C_tr_exponential_kernel_with_stop, 5, NULL},
  {"C_kernel_eval", (DL_FUNC) &C_kernel_eval, 5, NULL},
  {"C_llm_1D_beta", (DL_FUNC) &C_llm_1D_beta, 8, NULL},
  {"C_llm_1D_beta_perms", (DL_FUNC) &C_llm_1D_beta_perms, 11, NULL},
  {"C_predict_1D", (DL_FUNC) &C_predict_1D, 11, NULL},
  {"C_wpredict_1D", (DL_FUNC) &C_wpredict_1D, 11, NULL},
  {"C_llm_1D_fit_and_predict", (DL_FUNC) &C_llm_1D_fit_and_predict, 12, NULL},
  {"C_mllm_1D_fit_and_predict", (DL_FUNC) &C_mllm_1D_fit_and_predict, 12, NULL},
  {"C_llm_1D_fit_and_predict_BB_CrI", (DL_FUNC) &C_llm_1D_fit_and_predict_BB_CrI, 13, NULL},
  {"C_llm_1D_fit_and_predict_global_BB_CrI", (DL_FUNC) &C_llm_1D_fit_and_predict_global_BB_CrI, 13, NULL},
  {"C_llm_1D_fit_and_predict_global_BB", (DL_FUNC) &C_llm_1D_fit_and_predict_global_BB, 11, NULL},
  {"C_llm_1D_fit_and_predict_global_BB_qCrI", (DL_FUNC) &C_llm_1D_fit_and_predict_global_BB_qCrI, 13, NULL},
  {"C_loo_llm_1D", (DL_FUNC) &C_loo_llm_1D, 11, NULL},
  {"C_deg0_loo_llm_1D", (DL_FUNC) &C_deg0_loo_llm_1D, 7, NULL},
  {"C_cv_mae_1D", (DL_FUNC) &C_cv_mae_1D, 16, NULL},
  {"C_get_BB_Eyg", (DL_FUNC) &C_get_BB_Eyg, 19, NULL},
  {"C_get_Eyg_CrI", (DL_FUNC) &C_get_Eyg_CrI, 19, NULL},
  {"C_get_Eygs", (DL_FUNC) &C_get_Eygs, 17, NULL},
  {"C_get_bws", (DL_FUNC) &C_get_bws, 6, NULL},
  {"C_get_bws_with_minK_a", (DL_FUNC) &C_get_bws_with_minK_a, 6, NULL},
  {"C_columnwise_weighting", (DL_FUNC) &C_columnwise_weighting, 7, NULL},
  {"C_columnwise_eval", (DL_FUNC) &C_columnwise_eval, 5, NULL},
  {"C_columnwise_TS_norm", (DL_FUNC) &C_columnwise_TS_norm, 4, NULL},
  {"C_matrix_wmeans", (DL_FUNC) &C_matrix_wmeans, 9, NULL},
  {"C_columnwise_wmean", (DL_FUNC) &C_columnwise_wmean, 6, NULL},
  {"C_columnwise_wmean_BB", (DL_FUNC) &C_columnwise_wmean_BB, 7, NULL},
  {"C_columnwise_wmean_BB_qCrI", (DL_FUNC) &C_columnwise_wmean_BB_qCrI, 9, NULL},
  {"C_columnwise_wmean_BB_CrI_1", (DL_FUNC) &C_columnwise_wmean_BB_CrI_1, 8, NULL},
  {"C_columnwise_wmean_BB_CrI_2", (DL_FUNC) &C_columnwise_wmean_BB_CrI_2, 8, NULL},
  {"C_normalize_dist", (DL_FUNC) &C_normalize_dist, 7, NULL},
  {"C_pearson_wcor", (DL_FUNC) &C_pearson_wcor, 5, NULL},
  {"C_pearson_wcor_BB_qCrI", (DL_FUNC) &C_pearson_wcor_BB_qCrI, 10, NULL},
  {NULL, NULL, 0, NULL}
};

static const R_CallMethodDef callMethods[] = {
  {"S_generate_dirichlet_weights", (DL_FUNC) &S_generate_dirichlet_weights, 2},
  {"S_get_BB_Eyg_external", (DL_FUNC) &S_get_BB_Eyg_external, 18},
  {"S_magelog", (DL_FUNC) &S_magelog, 15},
  {"S_mabilog", (DL_FUNC) &S_mabilog, 14},
  {"S_mabilo_plus", (DL_FUNC) &S_mabilo_plus, 13},
  {"S_mabilo", (DL_FUNC) &S_mabilo, 11},
  {NULL, NULL, 0}
};

void R_init_malo(DllInfo *dll)
{
  R_registerRoutines(dll, cMethods, callMethods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, FALSE);
}
