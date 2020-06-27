# def calc_hv(pf_list, ref_point=(1.1, 1.1), f_normalization=None):
#     hv_indicator = get_performance_indicator("hv", ref_point=np.array([1.1, 1.1]))
#
#     hv_list = []
#     for pf in pf_list:
#         if f_normalization is not None:
#
#         hv_all_gen = []
#         for i, key in enumerate(hf.keys()):
#             gen_no = int(key[3:])
#             f_current_gen = np.array(hf[key]['F'])
#             # f_min_point = np.array([0, 0])
#             # f_max_point = np.array([50000, 15])
#             f_current_gen_normalized = (f_current_gen - f_min_point) / (f_max_point - f_min_point)
#             hv_current_gen = hv_indicator.calc(f_current_gen_normalized)
#             hv_all_gen.append([gen_no, hv_current_gen])
#
#         hv_all_gen = np.array(hv_all_gen)
#         hv_all_gen = hv_all_gen[np.argsort(hv_all_gen[:, 0]), :]
#         hf.close()
#
#     return hv_all_gen