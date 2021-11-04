import numpy as np
orgin_file =  r'.\000000201706_unique_test.txt'
writed_file =  r'.\000000201706_unique_test_rewrite.txt'
s = np.loadtxt(orgin_file)
print(f"    {s.shape=}")
s_unique = np.unique(s ,axis=0)
print(f"    {s_unique.shape=}")
np.savetxt(writed_file, s_unique,fmt="%f", delimiter=" ")
s_unique_loaded = np.loadtxt(writed_file)
print(f"{    s_unique_loaded.shape=}")
